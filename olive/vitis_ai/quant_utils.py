#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import numpy as np
import onnx
import os
import onnx.helper as helper
import onnxruntime as ort
from enum import Enum
from onnxruntime.quantization.quant_utils import quantize_data


class PowerOfTwoMethod(Enum):
    NonOverflow = 0
    MinMSE = 1


def get_pos_overflow(f_tensor, bit_width=8):
    """
    Obtain the fixed-point position values using the non overflow method with power of 2 scale.
    """
    x_min = f_tensor.min()
    x_max = f_tensor.max()
    #  Use 0.5 as a guard
    lower_bound = -2**(bit_width - 1) - 0.5
    upper_bound = 2**(bit_width - 1) - 0.5
    scale = max(x_min / lower_bound, x_max / upper_bound)
    if scale == 0:
        # Set pos to 127(max of uint8) for all zero
        return 127
    else:
        pos = np.floor(np.log2(1 / scale))
    return pos


def get_pos_min_mse(f_tensor, bit_width=8, pos_range=5):
    """
    Obtain the fixed-point position values using the min mse method with power of 2 scale.
    """
    pos_overflow = get_pos_overflow(f_tensor, bit_width)
    if pos_overflow == 127:
        # Set pos to 127(max of uint8) for all zero
        return 127

    pos_diffs = pos_overflow
    diff_min = float('inf')
    for i in range(pos_range):
        tmp_pos = pos_overflow + i
        q_tensor = vitis_quantize(f_tensor, tmp_pos, bit_width=bit_width)
        diff = np.sum((q_tensor - f_tensor)**2)
        if diff < diff_min:
            diff_min = diff
            pos_diffs = tmp_pos
    return pos_diffs


def vitis_quantize(f_tensor, pos, bit_width=8):
    """
    Quantize the tensor using the corresponding fixed-point position.
    """
    scale, lower_bound, upper_bound = get_bound_and_scale(pos)
    q_tensor = np.round(f_tensor / scale) * scale
    q_tensor = lower_bound * (q_tensor < lower_bound) \
                + q_tensor * (q_tensor >= lower_bound)
    q_tensor = upper_bound * (q_tensor > upper_bound) \
                + q_tensor * (q_tensor <= upper_bound)
    return q_tensor


def get_bound_and_scale(pos, bit_width=8):
    """
    Obtain the scale and bound corresponding to the fixed-point position.
    """
    scale = np.power(2., -pos)
    lower_bound = -np.power(2, bit_width - 1) * scale
    upper_bound = np.power(2, bit_width - 1) * scale - scale
    return scale, lower_bound, upper_bound


def vitis_quantize_data(data, bit_width=8, method="overflow"):
    """
    Quantize the input data using the PowerOfTwoMethod.
    """
    if method == PowerOfTwoMethod.NonOverflow:
        pos = get_pos_overflow(data)
    elif method == PowerOfTwoMethod.MinMSE:
        pos = get_pos_min_mse(data)

    scale, lower_bound, upper_bound = get_bound_and_scale(pos)
    zero_point = 0
    quantized_data = vitis_quantize(data, pos=pos)
    return lower_bound, upper_bound, zero_point, scale, quantized_data


def scale2pos(scale):
    """
    Obtain the fixed-point position corresponding to the scale.
    """
    return int(np.rint(-np.log2(scale)))


def pos2scale(pos):
    """
    Obtain the scale corresponding to the fixed-point position.
    """
    return float(np.power(2., -pos))


def get_exclude_nodes(model_path, input_nodes, output_nodes):
    """
    Return the nodes to be excluded based on the given input and output nodes.
    """
    model = onnx.load(model_path)

    name_list = []
    exclude_nodes = []

    for node in model.graph.node:
        name_list.append(node.name)

    if input_nodes:
        for name in input_nodes:
            input_node_name = name
            index = name_list.index(input_node_name)
            exclude_nodes_i = name_list[:index]
            exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_i))
            exclude_nodes = list(set(exclude_nodes) - set(input_nodes))

    if output_nodes:
        for name in output_nodes:
            output_node_name = name
            index = name_list.index(output_node_name)
            exclude_nodes_o = name_list[index+1:]
            exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_o))
            exclude_nodes = list(set(exclude_nodes) - set(output_nodes))

    return exclude_nodes
