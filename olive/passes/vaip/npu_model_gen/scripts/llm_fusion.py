##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import numpy
import argparse
import sys
import os
import onnx
import copy
import json
from pathlib import Path
from enum import Enum
from colorama import init, Fore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onnx_tools.graph import *
from onnx_tools.fusion import *
from onnx import helper, TensorProto
from scripts.helper import *
from scripts.extraction_utility import extract_subgraph


def create_patterns(input_model_path, dict):
    pattern_dict = {}
    m, g = loadmodel(input_model_path)

    for key in dict.keys():
        desc = create_descs_from_nodenames(g, dict[key])
        if key in pattern_dict.keys():
            pattern_dict[key].append(desc)
        else:
            pattern_dict[key] = [desc]

    return pattern_dict


def fuse_layers(input_model_path, output_model_path, patterns):
    m, g = loadmodel(input_model_path)

    for key in patterns.keys():
        # print(key)
        for fuse_pattern in patterns[key]:
            # print("Pattern Key: {}, Pattern Length: {}".format(key, len(fuse_pattern)))

            pattern = FusionPattern(fuse_pattern)
            subgraphs = pattern.search_pattern(g)

            # for each occurence of the pattern, fuse
            try:
                for nodes in subgraphs:
                    k = Graph(
                        g.get_onnxgraph_by_nodenames(nodes),
                        ModelConfig({}),
                    )
                    node_outputs = []

                    for n in nodes:
                        node_outputs.extend(k.nodemap[n].output)

                    g.fuse_subgraph_node_names(nodes, key, nodes[0], True)
            except Exception as e:
                print(e)

    for node_name in g.nodemap.keys():
        if g.nodemap[node_name].op_type == "SSMLP":
            move_weight = g.nodemap[node_name].input.pop()
            g.nodemap[node_name].input.insert(3, move_weight)

    m.mproto.graph.ClearField("node")

    nodes = []
    for name in g.nodemap.keys():
        nodes.append(g.nodemap[name].make_nodeproto())

    m.mproto.graph.node.extend(nodes)

    try:
        onnx.save_model(
            m.mproto,
            output_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
            size_threshold=1024,
            convert_attribute=False,
        )
        print("- Generated model at: ", output_model_path)
    except Exception as e:
        raise e


def fuse_main(
    input_model_path,
    output_model_path,
    key1,
    key2,
):
    # m, g = loadmodel(input_model_path)

    # # removing the empty inputs/outputs in the graph
    # for i in g.nodemap.keys():
    #     if g.nodemap[i].op_type == "MultiHeadAttention":
    #         g.nodemap[i].input.pop(3)
    #         g.nodemap[i].input.pop(3)
    #     if g.nodemap[i].op_type == "SkipSimplifiedLayerNormalization":
    #         list1 = []
    #         for j in g.nodemap[i].output:
    #             if j != "":
    #                 list1.append(j)
    #         g.nodemap[i].output = list1

    # # print("Length of nodemap: ",len(m.graph.nodemap))

    # nos = [
    #     "/model/layers.0/post_attention_layernorm/SkipLayerNorm",
    #     "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_fp32_",
    #     "/model/layers.0/mlp/gate_proj/MatMulNBits",
    #     "/model/layers.0/mlp/gate_proj/MatMulNBits_cast_bf16",
    #     "/model/layers.0/mlp/act_fn/Sigmoid_cast_fp32_0",
    #     "/model/layers.0/mlp/act_fn/Mul_cast_fp32_x",
    #     "/model/layers.0/mlp/act_fn/Sigmoid",
    #     "/model/layers.0/mlp/act_fn/Sigmoid_cast_bf16_0",
    #     "/model/layers.0/mlp/act_fn/Mul_cast_fp32_y",
    #     "/model/layers.0/mlp/act_fn/Mul",
    #     "/model/layers.0/mlp/act_fn/Mul_cast_bf16_",
    #     "/model/layers.0/mlp/Mul_cast_fp32_x",
    #     "/model/layers.0/mlp/up_proj/MatMulNBits_cast_fp32_",
    #     "/model/layers.0/mlp/up_proj/MatMulNBits",
    #     "/model/layers.0/mlp/up_proj/MatMulNBits_cast_bf16",
    #     "/model/layers.0/mlp/Mul_cast_fp32_y",
    #     "/model/layers.0/mlp/Mul",
    #     "/model/layers.0/mlp/Mul_cast_bf16_",
    #     "/model/layers.0/mlp/down_proj/MatMulNBits_cast_fp32_",
    #     "/model/layers.0/mlp/down_proj/MatMulNBits",
    #     "/model/layers.0/mlp/down_proj/MatMulNBits_cast_bf16",
    #     "/model/layers.1/input_layernorm/SkipLayerNorm",
    # ]

    # nos1 = [
    #     "/model/layers.0/attn/GroupQueryAttention",
    #     "/model/layers.0/attn/o_proj/MatMulNBits_cast_fp32_",
    #     "/model/layers.0/attn/o_proj/MatMulNBits",
    #     "/model/layers.0/attn/o_proj/MatMulNBits_cast_bf16",
    # ]

    # if key1 and key2:
    #     dictionary = {key1: nos, key2: nos1}
    # elif key1 == "SSMLP":
    #     dictionary = {key1: nos}
    # elif key1 == "GQO":
    #     dictionary = {key1: nos1}

    # patterns = create_patterns(input_model_path, dictionary)
    from .LLM_patterns import patterns
    if key2 == None and key1 == "SSMLP":
        patterns.pop("GQO")
    elif key2 == None and key1 == "GQO":
        patterns.pop("SSMLP")
        
    # elif key1 == "SSMLP":
    #     dictionary = {key1: nos}
    # elif key1 == "GQO":
    #     dictionary = {key1: nos1}
    
    # use the patterns to fuse any matched subgraph in the input_model
    fuse_layers(input_model_path, output_model_path, patterns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_model_path",
        metavar="I/P_path",
        help="path to model to be fused",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_model_path",
        metavar="O/P_path",
        help="path to store new fused model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--key1", help="name of the fused node", type=str, required=True
    )
    parser.add_argument("--key2", help="name of the fused node", type=str)
    args = parser.parse_args()

    fuse_main(
        args.input_model_path,
        args.output_model_path,
        args.key1,
        args.key2,
    )
