#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import onnx
from onnx import helper
from onnxsim import simplify

# facility code: used for counding target op in a onnx model
def onnx_contains_op_num(model_path: str, target_op_type: str) -> int:
    model = onnx.load(model_path)
    count = 0
    for node in model.graph.node:
        if node.op_type == target_op_type:
            count += 1
    return count


def mk_quant_de_quant_node(graph, quant_node_name, dequant_node_name, modify_node, idx):
    quant_node = next((node for node in graph.node if node.name == quant_node_name), None)
    dequant_node = next((node for node in graph.node if node.name == dequant_node_name), None)
    modify_node = next((node for node in graph.node if node.name == modify_node), None)

    new_quant_node_name = quant_node_name + "_sp_" + str(idx)
    new_quant_node_output = quant_node.output[0] + "_sp_" + str(idx)

    new_dequant_node_name = dequant_node_name + "_sp_" + str(idx)
    new_dequant_node_output = dequant_node.output[0] + "_sp_" + str(idx)

    new_quant_node2 = helper.make_node(
        op_type=quant_node.op_type,
        inputs=quant_node.input,
        outputs=[new_quant_node_output],
        name=new_quant_node_name,
        **{attr.name: helper.get_attribute_value(attr) for attr in quant_node.attribute}   # copy attribute
    )

    graph.node.append(new_quant_node2)

    new_dequant_input = [new_quant_node_output] + dequant_node.input[1:]
    new_dequant_node2 = helper.make_node(
        op_type=dequant_node.op_type,
        inputs=new_dequant_input,  # dequant_node.input
        outputs=[new_dequant_node_output],  # use new output
        name=new_dequant_node_name,
        **{attr.name: helper.get_attribute_value(attr) for attr in dequant_node.attribute}  # copy attribute
    )

    graph.node.append(new_dequant_node2)
    modify_node.input[1] = new_dequant_node_output
    return


if __name__ == '__main__':

    # NOTE user need to modify the onnx model path
    exported_onnx_model = "./quant_result/quark_model.onnx"
    if not os.path.exists(exported_onnx_model):
        raise FileNotFoundError("The file: {} not found".format(exported_onnx_model))

    # NOTE using this function to check the quantizer num
    onnx_contains_op_num(exported_onnx_model, "DequantizeLinear")

    quant_model = onnx.load(exported_onnx_model)
    model_simp, check = simplify(quant_model)
    onnx.save_model(model_simp, "./quant_result/sample_quark_model.onnx")  # NOTE modify the path

    model = model_simp
    graph = model.graph

    # NOTE following code is optional to use
    # -------- optional to use -------
    # The following parameters are only adapt for YOLO-NAS quantization result
    Quant_linear_1_name = "/original_model/fake_quantizer_172/QuantizeLinear"
    DeQuant_linear_1_name = "/original_model/fake_quantizer_172/DequantizeLinear"
    conv1_name = "/original_model/Conv"
    conv2_name = "/original_model/Conv_1"
    conv3_name = "/original_model/Conv_2"
    Quant_linear_2_name = '/original_model/fake_quantizer_211/QuantizeLinear'
    DeQuant_linear_2_name = '/original_model/fake_quantizer_211/DequantizeLinear'
    add_name = '/original_model/Add_20'

    # create a new node and copy the attribute of original node and give a new name
    mk_quant_de_quant_node(graph, Quant_linear_1_name, DeQuant_linear_1_name, conv2_name, 1)
    mk_quant_de_quant_node(graph, Quant_linear_1_name, DeQuant_linear_1_name, conv3_name, 2)
    mk_quant_de_quant_node(graph, Quant_linear_2_name, DeQuant_linear_2_name, add_name, 1)
    onnx.save(model, "./onnx_model/modified_sample_Quark_fx_qat.onnx")
    # -------- optional to use -------
