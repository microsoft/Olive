#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import sys

import numpy as np
import onnx
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.quant_utils import get_qmin_qmax_for_qType, quantize_nparray
from packaging import version

from olive.common.utils import IntEnumBase

# pylint: skip-file
# ruff: noqa


class PowerOfTwoMethod(IntEnumBase):
    NonOverflow = 0
    MinMSE = 1


def get_pos_overflow(f_tensor, bit_width=8):
    """
    Obtain the fixed-point position values using the non overflow method with power of 2 scale.
    """
    x_min = f_tensor.min()
    x_max = f_tensor.max()
    #  Use 0.5 as a guard
    lower_bound = -(2 ** (bit_width - 1)) - 0.5
    upper_bound = 2 ** (bit_width - 1) - 0.5
    scale = max(x_min / lower_bound, x_max / upper_bound)
    if scale == 0:
        # Set pos to 127(max of uint8) for all zero
        return 127
    else:
        pos = np.floor(np.log2(1 / scale))
        pos = 127 if pos > 127 else pos
        pos = -128 if pos < -128 else pos
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
    diff_min = float("inf")
    for i in range(pos_range):
        tmp_pos = pos_overflow + i
        q_tensor = vitis_quantize(f_tensor, tmp_pos, bit_width=bit_width)
        diff = np.sum((q_tensor - f_tensor) ** 2)
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
    q_tensor = lower_bound * (q_tensor < lower_bound) + q_tensor * (q_tensor >= lower_bound)
    q_tensor = upper_bound * (q_tensor > upper_bound) + q_tensor * (q_tensor <= upper_bound)
    return q_tensor


def get_bound_and_scale(pos, bit_width=8):
    """
    Obtain the scale and bound corresponding to the fixed-point position.
    """
    scale = np.power(2.0, -pos)
    lower_bound = -np.power(2, bit_width - 1) * scale
    upper_bound = np.power(2, bit_width - 1) * scale - scale
    return scale, lower_bound, upper_bound


def vitis_quantize_data(data, bit_width=8, method=PowerOfTwoMethod.NonOverflow):
    """
    Quantize the input data using the PowerOfTwoMethod.
    """
    if method == PowerOfTwoMethod.NonOverflow:
        pos = get_pos_overflow(data)
    elif method == PowerOfTwoMethod.MinMSE:
        pos = get_pos_min_mse(data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    scale, lower_bound, upper_bound = get_bound_and_scale(pos)
    zero_point = 0
    quantized_data = vitis_quantize(data, pos=pos)
    return lower_bound, upper_bound, zero_point, scale, quantized_data


def scale2pos(scale):
    """
    Obtain the fixed-point position corresponding to the scale.
    """
    if scale > 2**127:
        return 127
    if scale < 2 ** (-128):
        return -128
    return int(np.rint(-np.log2(scale)))


def pos2scale(pos):
    """
    Obtain the scale corresponding to the fixed-point position.
    """
    return float(np.power(2.0, -pos))


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
            index = name_list.index(output_node_name) + 1
            exclude_nodes_o = name_list[index:]
            exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_o))
            exclude_nodes = list(set(exclude_nodes) - set(output_nodes))

    return exclude_nodes


def get_annotate_output_name(model):
    """
    Traverse the floating-point model to find the names of all conv outputs.
    """
    annotate_output_name_list = []
    annotate_op_list = ["Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool"]

    for node in model.graph.node:
        if node.op_type in annotate_op_list:
            annotate_output_name_list.append(node.output[0])
    return annotate_output_name_list


def get_relu_name(model, input_list):
    """
    Find the corresponding pairs that form conv-relu through the output of conv,
    and return two dictionaries: conv-relu and relu-conv
    """
    relu_to_conv_output = {}
    for node in model.graph.node:
        if node.op_type == "Relu" and node.input[0] in input_list:
            relu_to_conv_output[node.name] = node.input[0]
        elif node.op_type == "LeakyRelu":
            for attr in node.attribute:
                if attr.name == "alpha":
                    if attr.f == 0.1 and node.input[0] in input_list:
                        relu_to_conv_output[node.name] = node.input[0]

    return relu_to_conv_output


def get_qdq_to_remove(model, relu_input):
    """
    Return two lists, one for 'dq' and one for 'q'.
    """
    q_nodes_to_remove = []
    dq_nodes_to_remove = []
    q_nodes_output_to_remove = []
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[0] in relu_input.values():
            q_nodes_to_remove.append(node)
            q_nodes_output_to_remove.append(node.output[0])
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and node.input[0] in q_nodes_output_to_remove:
            dq_nodes_to_remove.append(node)
    return dq_nodes_to_remove, q_nodes_to_remove


def remove_nodes(model, nodes_list):
    """
    Delete nodes according to the nodes in the list
    """
    for node in nodes_list:
        model.graph.node.remove(node)
    return model


def convert_relu_input_to_annotate_output(model, relu_to_conv_output):
    """
    Modify the input of ReLU to the output of annotate op, and delete QDQ
    """
    for node in model.graph.node:
        if node.name in relu_to_conv_output.keys():
            node.input[0] = relu_to_conv_output[node.name]
    return model


def compute_scale_zp_pof2s(rmin, rmax, qmin, qmax, symmetric=False):
    """Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :return: zero and scale [z, s]

    """

    if qmin > 0 or qmax < 0:
        raise ValueError(f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}")

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    if is_ort_version_below_1_17():
        rmin = min(rmin, 0)
        rmax = max(rmax, 0)

        if rmin == float("-inf"):
            rmin = -sys.float_info.max / 2
        if rmax == float("inf"):
            rmax = sys.float_info.max / 2

        if symmetric:
            absmax = max(abs(rmin), abs(rmax))
            rmin = -absmax
            rmax = +absmax

        scale = (rmax - rmin) / float(qmax - qmin)

        pos = scale2pos(scale)
        pof2_scale = pos2scale(pos)

        if pof2_scale < np.finfo(np.float32).tiny:
            pof2_scale = 1.0
            zero_point = 0
        else:
            zero_point = round(qmin - rmin / pof2_scale)
        if symmetric:
            zero_point = 0
    else:
        # hasattr(rmin, "dtype") is True for onnxruntime>=1.17
        rmin_dtype = rmin.dtype if hasattr(rmin, "dtype") else np.float32

        rmin = np.minimum(rmin, np.array(0, dtype=rmin_dtype))
        rmax = np.maximum(rmax, np.array(0, dtype=rmin_dtype))

        # Ensure that rmax-rmin is less than or equal to sys.float_info.max
        if rmin == float("-inf"):
            rmin = -np.finfo(rmin_dtype).max / 2
        if rmax == float("inf"):
            rmax = np.finfo(rmin_dtype).max / 2

        if symmetric:
            absmax = np.maximum(np.abs(rmin), np.abs(rmax))
            rmin = -absmax
            rmax = +absmax

        assert qmin <= qmax, f"qmin={rmin} > qmax={rmax}"
        dr = np.array(rmax - rmin, dtype=np.float64)
        dq = np.array(qmax, dtype=np.float64) - np.array(qmin, dtype=np.float64)
        scale = np.array(dr / dq)

        pos = scale2pos(scale)
        pof2_scale = pos2scale(pos)

        has_dtype = hasattr(qmin, "dtype")  # True for onnxruntime>=1.17

        if pof2_scale < np.finfo(rmax.dtype).tiny:
            pof2_scale = np.array(1.0, dtype=rmax.dtype)
            zero_point = np.array(0, dtype=qmin.dtype) if has_dtype else 0
        else:
            pof2_scale = np.array(pof2_scale, dtype=rmin_dtype)
            zero_point = (
                np.array(np.round(qmin - rmin / pof2_scale), dtype=qmin.dtype)
                if has_dtype
                else int(round(qmin - rmin / pof2_scale))
            )
        if symmetric:
            zero_point = np.array(0, dtype=qmin.dtype) if has_dtype else 0
    return [zero_point, pof2_scale]


def quantize_zero_point(rmin, qmin, qmax, symmetric, scale):
    if qmin > 0 or qmax < 0:
        raise ValueError(f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}")

    rmin = min(rmin, 0)

    if symmetric:
        return 0 if is_ort_version_below_1_17() else np.array(0, dtype=qmin.dtype)

    if is_ort_version_below_1_17():
        scale_min = np.finfo(np.float32).tiny
        pof2_scale = scale
    else:
        scale_min = np.finfo(rmin.dtype).tiny
        pof2_scale = np.array(scale, dtype=rmin.dtype)

    if pof2_scale < scale_min:
        zero_point = 0
    else:
        zero_point = round(qmin - rmin / pof2_scale)

    return int(zero_point) if is_ort_version_below_1_17() else np.array(zero_point, dtype=qmin.dtype)


def dequantize_data(data, scale, zero_point):
    data = data.astype(np.float32)
    deq_arr = (data - zero_point) * scale
    return deq_arr.astype(np.float32) if is_ort_version_below_1_17() else deq_arr.astype(scale.dtype)


def quantize_data_pof2s(data, qType, symmetric, reduce_range=False, method=PowerOfTwoMethod.NonOverflow, pos_range=5):
    """
    :param data: data to quantize
    :param qType: data type to quantize to. Supported types UINT8 and INT8
    :param symmetric: whether symmetric quantization is used or not. This is applied to INT8.
    :return: minimum, maximum, zero point, scale, and quantized weights

    To pack weights, we compute a linear transformation

    - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
    - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
        `m = max(abs(rmin), abs(rmax))`

    and add necessary intermediate nodes to transform quantized weight to full weight using the equation

    :math:`r = S(q-z)`, where

    - *r*: real original value
    - *q*: quantized value
    - *S*: scale
    - *z*: zero point
    """
    if is_ort_version_below_1_17():
        rmin = 0
        rmax = 0
        if isinstance(data, np.ndarray):
            rmin = data.min()
            rmax = data.max()

        elif isinstance(data, list) and len(data):
            rmin = min(data)
            rmax = max(data)
        data = np.asarray(data)
    else:
        assert data.dtype in {
            np.float16,
            np.float32,
            np.dtype("float16"),
            np.dtype("float32"),
        }, f"Unexpected dtype {data.dtype!r}"
        rmin = data.min()
        rmax = data.max()
        assert rmin.dtype in {
            np.float16,
            np.float32,
            np.dtype("float16"),
            np.dtype("float32"),
        }, f"Unexpected dtype {rmin.dtype!r}"
        assert rmin.dtype == rmax.dtype

    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)
    zero_point, scale = compute_scale_zp_pof2s(rmin, rmax, qmin, qmax, symmetric)

    quantized_data = quantize_nparray(qType, data, scale, zero_point)

    if method == PowerOfTwoMethod.NonOverflow:
        return rmin, rmax, zero_point, scale, quantized_data

    elif method == PowerOfTwoMethod.MinMSE:
        rmin_dtype = rmin.dtype
        scale_mse = scale
        zp_mse = zero_point
        quantized_data_mse = quantized_data
        diff_min = float("inf")
        for i in range(pos_range):
            new_scale = pos2scale(scale2pos(scale) + i)
            rmin = min((qmin - zero_point) * new_scale, 0)
            if not is_ort_version_below_1_17():
                new_scale = np.array(new_scale, dtype=rmin_dtype)
                rmin = np.array(rmin, dtype=rmin_dtype)
            new_zero_point = quantize_zero_point(rmin, qmin, qmax, symmetric, new_scale)

            new_quantized_data = quantize_nparray(qType, data, new_scale, new_zero_point)
            diff = np.sum((dequantize_data(new_quantized_data, new_scale, new_zero_point) - data) ** 2)
            if diff < diff_min:
                diff_min = diff
                scale_mse = new_scale
                zp_mse = new_zero_point
                quantized_data_mse = new_quantized_data

        rmin_mse = (qmin - zp_mse) * scale_mse
        rmax_mse = (qmax - zp_mse) * scale_mse
        if not is_ort_version_below_1_17():
            rmin_mse = np.array(rmin_mse, dtype=rmin_dtype)
            rmax_mse = np.array(rmax_mse, dtype=rmax.dtype)

        return rmin_mse, rmax_mse, zp_mse, scale_mse, quantized_data_mse
    else:
        raise ValueError(f"Unsupported method: {method}")


def is_ort_version_below_1_16():
    """
    This function checks whether the current version of ONNX Runtime (ORT) is below 1.16.0.

    Returns:
        True if the current ORT version is less than 1.16.0, False otherwise.
    """
    return version.parse(OrtVersion) < version.parse("1.16.0")


def is_ort_version_below_1_17():
    """
    This function checks whether the current version of ONNX Runtime (ORT) is below 1.17.0.

    Returns:
        True if the current ORT version is less than 1.17.0, False otherwise.
    """
    return version.parse(OrtVersion) < version.parse("1.17.0")


def is_ort_version_below_1_18():
    """
    This function checks whether the current version of ONNX Runtime (ORT) is below 1.18.0.

    Returns:
        True if the current ORT version is less than 1.18.0, False otherwise.
    """
    return version.parse(OrtVersion) < version.parse("1.18.0")
