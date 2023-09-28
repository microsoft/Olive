# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict

import numpy as np
import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class ModelOptimizer:
    def __init__(self, source_model_path):
        self.source_model_path = str(source_model_path)
        self.prepare()

    def prepare(self):
        from onnxruntime.transformers.onnx_model import OnnxModel as TransformersOnnxModel

        self.model = onnx.load(self.source_model_path)
        self.onnx_model = TransformersOnnxModel(self.model)
        self.graph = self.onnx_model.graph()

        node_idx = 0
        self.node_name2module = {}

        for node in self.graph.node:
            if node.name == "":
                node.name = str(node.op_type) + str(node_idx)
            node_idx += 1
            self.node_name2module[node.name] = [node, node_idx]

    def optimize(self):
        self.fuse_transpose_qat()

    def fuse_transpose_qat(self):
        for node_name in self.node_name2module:
            node = self.node_name2module[node_name][0]
            node_index = self.node_name2module[node_name][1]
            if node.op_type == "Transpose":
                if "DequantizeLinear" in node.input[0]:
                    dequant_node_name = node.input[0][:-9]
                    new_dequant_node_output = node.output[0]
                    dequant_node = self.node_name2module[dequant_node_name][0]
                    x_node = self.node_name2module[dequant_node.input[0][:-9]][0]
                    x_scale_node = self.node_name2module[dequant_node.input[1][:-9]][0]
                    x_zero_point_node = self.node_name2module[dequant_node.input[2][:-9]][0]

                    x_val = self.onnx_model.get_constant_value(dequant_node.input[0])
                    new_x_val = np.transpose(x_val, axes=(1, 0))
                    x_scale_val = self.onnx_model.get_constant_value(dequant_node.input[1])
                    x_zero_point_val = self.onnx_model.get_constant_value(dequant_node.input[2])

                    self.remove_nodes(self.graph, [node, dequant_node, x_node, x_scale_node, x_zero_point_node])
                    new_dequant, x, x_scale, x_zero_point = self.create_dequantizelinear_node(
                        new_x_val, x_scale_val, x_zero_point_val, new_dequant_node_output, node_index
                    )
                    self.add_nodes(self.graph, [new_dequant, x, x_scale, x_zero_point], node_index)

    def create_dequantizelinear_node(self, x_val, x_scale_val, x_zero_point_val, outputs, node_name_suffix):
        x_tensor = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.DataType.INT8,
            dims=x_val.shape,
            vals=x_val.flatten().tobytes(),
            raw=True,
        )

        x_scale_tensor = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.DataType.FLOAT,
            dims=[1],
            vals=x_scale_val.tobytes(),
            raw=True,
        )

        x_zero_point_tensor = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.DataType.INT8,
            dims=[1],
            vals=x_zero_point_val.tobytes(),
            raw=True,
        )

        x = onnx.helper.make_node("Constant", inputs=[], outputs=["x_" + str(node_name_suffix)], value=x_tensor)
        x_scale = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["x_scale_" + str(node_name_suffix)], value=x_scale_tensor
        )
        x_zero_point = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["x_zero_point_" + str(node_name_suffix)], value=x_zero_point_tensor
        )

        dequant_node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[
                "x_" + str(node_name_suffix),
                "x_scale_" + str(node_name_suffix),
                "x_zero_point_" + str(node_name_suffix),
            ],
            outputs=[outputs],
        )
        return dequant_node, x, x_scale, x_zero_point

    def remove_nodes(self, graph, nodes_list):
        for node in nodes_list:
            graph.node.remove(node)

    def add_nodes(self, graph, nodes_list, node_index):
        for node in nodes_list:
            graph.node.insert(node_index, node)


class OnnxModelOptimizer(Pass):
    """Optimize ONNX model by fusing nodes."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        output_model_path = ONNXModel.resolve_path(output_model_path)

        # optimize model
        model_optimizer = ModelOptimizer(model.model_path)
        model_optimizer.optimize()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(model_optimizer.model, output_model_path, config)
