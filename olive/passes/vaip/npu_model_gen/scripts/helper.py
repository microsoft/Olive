##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy as np
from collections import OrderedDict
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from onnx_tools.graph import *
from onnx_tools.fusion import *
from onnx_tools.model import *
import onnx
import copy
import json


# from cal_coeff_utils import *
import csv
from colorama import init, Fore

init(autoreset=True)


def duplicate_layer(m, g, node_optype, save=False):
    """
    Duplicate a layer with multiple outputs, to facilitate fusion at a node with multiple outputs
    """

    node_keys = list(g.nodemap.keys())

    for node_name in node_keys:
        if (g.nodemap[node_name].op_type == node_optype) and (
            len(g.nodemap[node_name].nextnodes) > 1
        ):
            node = g.nodemap[node_name]

            orig_out_tensor = g.tensormap[node.output[0]]
            for i in range(1, len(node.nextnodes)):
                """
                1.Create new node
                2.new node's next node will be one of the next nodes
                3. Add new node in nodemap
                """

                new_node = copy.copy(node)
                new_node.name = node_name + "__" + str(i)
                new_node.nextnodes = [node.nextnodes[i]]
                if new_node.name not in g.nodemap.keys():
                    g.nodemap[new_node.name] = new_node
                """
                    4. Create output tensor
                    5. add it to tensormap
                    6. add output tensor name in new_node's output
                    """
                new_output_tensor = copy.copy(orig_out_tensor)
                new_output_tensor.name = orig_out_tensor.name + "_" + str(i)
                new_node.output = [new_output_tensor.name]
                """
                    7. Add tensor in tensormap
                    8. update preoducedby[new_tensor] with new_node
                    9. update consumedby[new_tensor] with one of the next nodes (ith node's name)
                    10. update the corresponding input of the consumer node with the created output tensor
                    """
                if new_output_tensor.name not in g.tensormap.keys():
                    g.tensormap[new_output_tensor.name] = new_output_tensor

                if new_output_tensor.name not in g.producedby:
                    g.producedby[new_output_tensor.name] = [new_node.name]
                # print(node.nextnodes[i].name)

                if new_output_tensor.name not in g.consumedby:
                    g.consumedby[new_output_tensor.name] = [node.nextnodes[i].name]

                    con_node = g.nodemap[node.nextnodes[i].name]
                    for j in range(len(con_node.input)):
                        if (
                            con_node.input[j] == node.output[0]
                        ):  # check old node's output
                            con_node.input[j] = new_node.output[
                                0
                            ]  # update new node's output
                            # new node's output consumed by update
                """
                    11. Update the consumed by of input tensor of the new_node ( currently it has the old node only )
                    """
                input_tensor_to_orig_node = node.input[0]

                g.consumedby[input_tensor_to_orig_node].append(new_node.name)
                """
                    12. update the prevnode's nextnodes
                    """
                if node.prevnodes:
                    prevnode = node.prevnodes[0]

                    prevnode.nextnodes.extend([new_node])

            zerothnextnode = node.nextnodes[0]
            node.nextnodes = [zerothnextnode]
            node.name = node_name

    g.graph_reorder_nodes()
    if save:
        g.save_model("PSF_v1.0_QReshape_dup.onnx", rawmodel=m.mproto)
    return g


def change_output_dtype(g):
    """
    Change the data type of output of Quantize and Dequantize layers according to zp and scale
    """
    nodes = g.nodemap.keys()
    for node_name in nodes:
        node = g.nodemap[node_name]
        if node.op_type == "QuantizeLinear":
            for input in node.input:
                if "z_p" in input or "zero_point" in input:
                    data_type = g.tensormap[input].numpy.dtype
                else:
                    data_type = np.int8
            g.tensormap[node.output[0]].dtype = data_type
        if node.op_type == "DequantizeLinear":
            for input in node.input:
                # if node.prevnodes==[] and g.tensormap[input].dtype!=np.int8 and 'scale' not in input and 'zero_point' not in input:
                #     g.tensormap[input].numpy=g.tensormap[input].numpy.astype(np.int8)

                # if "zp" in input or "zero_point" in input:
                #     data_type = g.tensormap[input].dtype
                # g.tensormap[input].numpy=g.tensormap[input].numpy.astype(np.int8)
                if "scale" in input:
                    data_type = g.tensormap[input].dtype
                else:
                    data_type = np.float32
            g.tensormap[node.output[0]].dtype = data_type
    return g


def loadmodel(input_path):
    mcfg = {
        "constant_folding": False,
        "node_rename": False,
        "if_fixed_branch": None,
        "fixed_topk": 0,
        "verbose": False,
    }
    m = Model(input_path, mcfg)
    g = m.graph
    g.graph_reorder_nodes()
    return m, g


def count_ops(g):
    # Utility to count op_types
    # Takes onnx_tool graph object
    # should load the model using loadmodel and pass g to this function
    # Return a dictionary
    op_count_dictionary = {}
    for node in g.nodemap:
        if g.nodemap[node].op_type in op_count_dictionary:
            op_count_dictionary[g.nodemap[node].op_type] += 1
        else:
            op_count_dictionary[g.nodemap[node].op_type] = 1
    return op_count_dictionary


def get_node_names(g):
    return g.nodemap.keys()
