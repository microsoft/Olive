# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import numpy
import onnx
import os
import pprint

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from onnxruntime.transformers.onnx_model import OnnxModel
from typing import Any, Dict


_domain = 'com.microsoft'
_json_separators = (',', ': ')
_expert_pattern = [
  'Slice', 'Concat', 'MatMul', 'Cast', 'Relu', 'MatMul', 'Cast',
  'Slice', 'Mul', 'Div', 'Add', 'Gather', 'Shape', 'Reshape', 'Reshape'
]


def _dump_graph(node: Message, filepath: str):
    with open(filepath, 'wt') as strm:
        json.dump(MessageToDict(node), fp=strm, indent=2, sort_keys=True, separators=_json_separators)
        strm.flush()


def _iterate_experts(model: OnnxModel, producers: dict):
  for node in model.get_nodes_by_op_type(_expert_pattern[0]):
      path = model.match_parent_path(node, _expert_pattern[1:], output_name_to_node=producers)
      if path:
          path.reverse()
          path.append(node)
          yield path


def _create_node_name(model: OnnxModel, op_type: str, prefix_a: str, prefix_b: str):
    prefix = ''
    last_slash = -1
    for i in range(0, min(len(prefix_a), len(prefix_b))):
        if prefix_a[i] == prefix_b[i]:
            prefix += prefix_a[i]
            if prefix_a[i] == '/':
                last_slash = i
        else:
            break

    if last_slash > 0:
        prefix = prefix[:last_slash+1]

    if not prefix.endswith('/'):
        prefix += '/'
    prefix += op_type

    return model.create_node_name(op_type, prefix)


def _create_ranked_model(model: OnnxModel, nodes: dict, producers: dict,
                         consumers: dict, experts: list, world_size: int):
    for expert in experts:
        div_node = nodes[expert[5]]
        div_node_oname = div_node.output[0]
        mul_nodes = [n for n in consumers[div_node_oname] if n.op_type == 'Mul']

        reshape_node = nodes[expert[1]]
        reshape_node_oname = reshape_node.output[0]
        slice_nodes = [n for n in consumers[reshape_node_oname] if n.op_type == 'Slice']

        concat_node = nodes[expert[13]]
        matmul_nodes = [producers[iname] for iname in concat_node.input]

        num_experts = len(consumers[div_node_oname])
        experts_per_rank = int(num_experts / world_size)

        for i in range(experts_per_rank, num_experts):
            mul_nodes[i].input.remove(div_node_oname)

            for iname in list(slice_nodes[i].input):
                slice_nodes[i].input.remove(iname)

            for oname in matmul_nodes[i].output:
                concat_node.input.remove(oname)


def _insert_commop_nodes(model: OnnxModel, nodes: dict, experts: list, world_size: int):
    for expert in experts:
        for i, j in [(0, 1), (-2, -1)]:
          prev_node = nodes[expert[i]]
          next_node = nodes[expert[j]]

          node_name = _create_node_name(model, 'AllToAll', prev_node.name, next_node.name)
          output_name = node_name + '_output_0'
          alltoall = onnx.helper.make_node(
              'AllToAll', prev_node.output, [output_name],
              name=node_name, domain=_domain, group_size=world_size)

          model.add_node(alltoall)
          OnnxModel.replace_node_input(next_node, prev_node.output[0], output_name)


def _replace_constant_value(constant: Message, value: int):
    for attr in constant.attribute:
        if attr.name == 'value':
            attr.CopyFrom(onnx.helper.make_attribute(
                attr.name,
                onnx.numpy_helper.from_array(numpy.array([value], numpy.int64)),
                attr.doc_string))
            return


def _fix_shapes(model: OnnxModel, nodes: dict, producers: dict, consumers: dict, experts: list, world_size: int):
    for expert in experts:
        div_node = nodes[expert[5]]
        div_node_iname = div_node.input[1]
        div_node_oname = div_node.output[0]

        num_experts = len(consumers[div_node_oname])
        experts_per_rank = num_experts / world_size
        _replace_constant_value(producers[div_node_iname], experts_per_rank)

        add_node = nodes[expert[4]]
        add_node_iname = add_node.input[1]
        _replace_constant_value(producers[add_node_iname], experts_per_rank - 1)

        reshape_node = nodes[expert[1]]
        for parent in model.get_parents(reshape_node, producers):
            if parent.op_type == 'Concat':
                concat_iname_0 = parent.input[0]
                _replace_constant_value(producers[concat_iname_0], int(num_experts / experts_per_rank))

                concat_iname_1 = parent.input[1]
                _replace_constant_value(producers[concat_iname_1], experts_per_rank)

                break


def run(world_size: int, input_filepath: str, output_dirpath: str, debug: bool=False):
    basename = os.path.splitext(os.path.basename(input_filepath))[0]

    model = OnnxModel(onnx.load_model(input_filepath))
    producers = model.output_name_to_node()
    consumers = model.input_name_to_nodes()

    if debug:
        OnnxModel.graph_topological_sort(model.graph)
        _dump_graph(model.model, os.path.join(output_dirpath, f'{basename}.graph'))

    experts = []
    for expert in _iterate_experts(model, producers):
        if debug:
            pprint.pprint([n.name for n in expert])

        div_node = expert[5]
        div_node_oname = div_node.output[0]
        if len(consumers[div_node_oname]) % world_size != 0:
            raise f'Number of expert paths on node "{div_node.name}" should be a multiple of ' + \
                    f'input world-size={world_size} but found {len(consumers[div_node_oname])}'

        experts.append([n.name for n in expert])

    output_filepaths = []
    for rank in range(0, world_size):
        rank_model = OnnxModel(onnx.load_model(input_filepath))
        producers = rank_model.output_name_to_node()
        consumers = rank_model.input_name_to_nodes()
        nodes = { node.name : node for node in rank_model.nodes() }

        _create_ranked_model(rank_model, nodes, producers, consumers, experts, world_size)
        _insert_commop_nodes(rank_model, nodes, experts, world_size)
        _fix_shapes(model, nodes, producers, consumers, experts, world_size)

        rank_model.prune_graph()
        output_filepath = os.path.join(output_dirpath, f'{basename}_{rank:02d}.onnx')
        rank_model.save_model_to_file(output_filepath)
        onnx.checker.check_model(rank_model.model)
        output_filepaths.append(output_filepath)

        if debug:
            _dump_graph(model.model, os.path.join(output_dirpath, f'{basename}_{rank:02d}.graph'))

    return output_filepaths


class OrtMoEExpertsDistributor(Pass):
    """
    Split the input model (and insert necessary communication operations)
    to prepare for distributed inferencing.
    """

    @staticmethod
    def _default_config() -> Dict[str, Dict[str, Any]]:
        return {
            "world_size": PassConfigParam(
                type_=int,
                default=2,
                required=True,
                description=(
                    "Number of GPU nodes to distribute the model for. Must be greater than 1."
                ),
            ),
        }


    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        output_filepaths = run(config['world_size'], model.model_path, output_model_path)
        return ONNXModel(output_model_path, model.name)
