# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from olive.constants import MSFT_DOMAIN, OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.hqq_quantization import OnnxHqqQuantization


class TestHQQQuantization:
    @pytest.fixture
    def matmul_model_path(self, tmp_path):
        """Create a simple ONNX model with a MatMul op and save it to a temporary file."""
        # Create input tensor
        input_shape = [1, 64]
        weight_shape = [64, 128]
        weight_tensor = np.random.randn(*weight_shape).astype(np.float32)

        # Create model
        input_name = "input"
        output_name = "output"
        weight_name = "weight"

        input_tensor_proto = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        output_tensor_proto = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 128])
        weight_tensor_proto = onnx.helper.make_tensor(
            name=weight_name, data_type=onnx.TensorProto.FLOAT, dims=weight_shape, vals=weight_tensor.flatten().tolist()
        )

        # Create MatMul node
        matmul_node = onnx.helper.make_node(
            str(OpType.MatMul), inputs=[input_name, weight_name], outputs=[output_name], name="MatMul_Node"
        )

        graph_def = onnx.helper.make_graph(
            nodes=[matmul_node],
            name="test-model",
            inputs=[input_tensor_proto],
            outputs=[output_tensor_proto],
            initializer=[weight_tensor_proto],
        )

        # Create model
        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13

        # Save model
        model_path = tmp_path / "matmul_model.onnx"
        onnx.save(model_def, str(model_path))
        return model_path

    def test_hqq_quantization_pass(self, matmul_model_path, tmp_path):
        # Setup
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"block_size": 128}
        p = create_pass_from_dict(
            OnnxHqqQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        # Execute
        output_path = tmp_path / "quantized_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        # Assert
        assert os.path.exists(quantized_model.model_path)

        # Load the quantized model and check for MatMulNBits nodes
        ir_model = ir.load(quantized_model.model_path)

        found_matmul_nbits = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == OpType.MatMulNBits:
                found_matmul_nbits = True
                assert node.domain == MSFT_DOMAIN
                assert node.attributes.get_int("bits") == 4
                assert node.attributes.get_int("block_size") == pass_config["block_size"]
                break

        assert found_matmul_nbits, "No MatMulNBits node found in quantized model"
