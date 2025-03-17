##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import onnx
import argparse
import numpy as np
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..\\modules"))
)
import _ExportConstants


i_map = {}


def preprocess_matmulnbits_weights(node, packed_const_index=5):
    global i_map

    k = onnx.helper.get_node_attr_value(node, "K")
    n = onnx.helper.get_node_attr_value(node, "N")
    block_size = onnx.helper.get_node_attr_value(node, "block_size")
    bits = onnx.helper.get_node_attr_value(node, "bits")
    start_index = 1
    weight_tensor = i_map[node.input[start_index]]
    scales_tensor = i_map[node.input[start_index + 1]]
    zeros_tensor = i_map[node.input[start_index + 2]]

    bias_enable = False
    bias_shape = (n, 1)
    if node.input[start_index + 3] == "":
        bias = np.zeros(bias_shape).astype(np.float32)
    else:
        bias_tensor = i_map[node.input[start_index + 3]]  # g_idx, then bias
        bias = onnx.numpy_helper.to_array(bias_tensor)
        bias_enable = True

    weight = onnx.numpy_helper.to_array(weight_tensor)
    scales = onnx.numpy_helper.to_array(scales_tensor)
    zero_point = onnx.numpy_helper.to_array(zeros_tensor)
    asymmetric_quant = True

    packed_weight: np.ndarray = _ExportConstants.matmulnbits_pack_const_float32(
        weight.astype(np.uint8),
        bias.astype(np.float32),
        scales.astype(np.float32),
        zero_point.astype(np.uint8),
        k,
        n,
        block_size,
        bias_enable,
        asymmetric_quant,
    )
    packed_weight_name = node.name + "_packed"

    packed_weight_tensor = onnx.helper.make_tensor(
        packed_weight_name,
        onnx.TensorProto.UINT8,
        packed_weight.shape,
        packed_weight.tobytes(),
        True,
    )

    return packed_weight_tensor


def pack_main(model_path, output, fuse):
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..\\modules"))
    )
    import _ExportConstants

    model = onnx.load(model_path)
    if fuse:
        location = "pack_model.data"
    else:
        location = "model.data"

    for init in model.graph.initializer:
        i_map[init.name] = init

    for node in model.graph.node:
        if node.op_type == "MatMulNBits":

            packed_wts = preprocess_matmulnbits_weights(node)

            total_pops = len(node.input) - 1

            for i in range(total_pops):
                if (
                    node.input[total_pops - i] != ""
                    and node.input[total_pops - i] in i_map
                ):
                    i_map.pop(node.input[total_pops - i])
                node.input.pop()
            for i in range(4):
                node.input.append("")
            node.input.append(packed_wts.name)
            # node.input[-1] = packed_wts.name

            i_map[packed_wts.name] = packed_wts

    i_map_list = list(i_map.values())
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(i_map_list)

    onnx.save_model(
        model,
        output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=1024,
        convert_attribute=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Input model", type=str, required=True)
    parser.add_argument("--output", help="Output path", type=str, required=True)
    args = parser.parse_args()
    model_path = args.model

    # load the model
    # m = onnx.load(model_path)
    output = args.output
    fuse = False
    pack_main(model_path, output)

    # for init in model.graph.initializer:
    #     i_map[init.name] = init

    # for node in model.graph.node:
    #     if node.op_type == "MatMulNBits":

    #         packed_wts = preprocess_matmulnbits_weights(node)

    #         total_pops = len(node.input) - 1

    #         for i in range(total_pops):
    #             if node.input[total_pops-i]!="": i_map.pop(node.input[total_pops-i])
    #             node.input.pop()
    #         for i in range(4):
    #             node.input.append("")
    #         node.input.append(packed_wts.name)
    #         # node.input[-1] = packed_wts.name

    #         i_map[packed_wts.name] = packed_wts

    # i_map_list = list(i_map.values())
    # model.graph.ClearField("initializer")
    # model.graph.initializer.extend(i_map_list)

    # onnx.save_model(
    #     model,
    #     args.output,
    #     save_as_external_data=True,
    #     all_tensors_to_one_file=True,
    #     location="model.data",
    #     size_threshold=1024,
    #     convert_attribute=False,
    # )
