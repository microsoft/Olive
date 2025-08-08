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
from olive.passes.onnx.rtn_quantization import OnnxBlockWiseRtnQuantization


class TestRTNQuantization:
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

    @pytest.fixture
    def gather_model_path(self, tmp_path):
        """Create a simple ONNX model with a Gather op and save it to a temporary file."""
        # Create input tensor and weights
        data_shape = [100, 64]  # vocabulary size, embedding dimension
        indices_shape = [1, 10]  # batch size, sequence length
        data_tensor = np.random.randn(*data_shape).astype(np.float32)

        # Create model
        data_name = "data"
        indices_name = "indices"
        output_name = "output"

        data_tensor_proto = onnx.helper.make_tensor(
            name=data_name, data_type=onnx.TensorProto.FLOAT, dims=data_shape, vals=data_tensor.flatten().tolist()
        )
        indices_tensor_proto = onnx.helper.make_tensor_value_info(indices_name, onnx.TensorProto.INT64, indices_shape)
        output_tensor_proto = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 10, 64])

        # Create Gather node
        gather_node = onnx.helper.make_node(
            str(OpType.Gather), inputs=[data_name, indices_name], outputs=[output_name], name="Gather_Node"
        )

        graph_def = onnx.helper.make_graph(
            nodes=[gather_node],
            name="test-gather-model",
            inputs=[indices_tensor_proto],
            outputs=[output_tensor_proto],
            initializer=[data_tensor_proto],
        )

        # Create model
        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13

        # Save model
        model_path = tmp_path / "gather_model.onnx"
        onnx.save(model_def, str(model_path))
        return model_path

    @pytest.mark.parametrize("is_symmetric", [True, False])
    def test_rtn_quantization_pass_matmul(self, matmul_model_path, tmp_path, is_symmetric):
        # Setup
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"bits": 4, "block_size": 128, "axis": 0, "is_symmetric": is_symmetric}
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        # Execute
        output_path = tmp_path / "quantized_matmul_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        # Assert
        assert os.path.exists(quantized_model.model_path)

        # Load the quantized model and check for MatMulNBits nodes
        quantized_onnx = onnx.load(quantized_model.model_path)

        found_matmul_nbits = False
        for node in quantized_onnx.graph.node:
            if node.op_type == str(OpType.MatMulNBits):
                found_matmul_nbits = True
                break

        assert found_matmul_nbits, "No MatMulNBits node found in quantized model"

    @pytest.mark.parametrize("is_symmetric", [True, False])
    def test_rtn_quantization_pass_gather(self, gather_model_path, tmp_path, is_symmetric):
        # Setup
        olive_model = ONNXModelHandler(model_path=str(gather_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"bits": 4, "block_size": 128, "axis": 0, "is_symmetric": is_symmetric}
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        # Execute
        output_path = tmp_path / "quantized_gather_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        # Assert
        assert os.path.exists(quantized_model.model_path)

        # Load the quantized model and check for GatherBlockQuantized nodes
        ir_model = ir.load(quantized_model.model_path)

        # Assert
        found_gather_block_quantized = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == str(OpType.GatherBlockQuantized):
                found_gather_block_quantized = True
                assert node.domain == MSFT_DOMAIN
                assert any(
                    attr.name == "block_size" and attr.value == pass_config["block_size"]
                    for attr in node.attributes.values()
                )
                assert any(
                    attr.name == "quantize_axis" and attr.value == pass_config["axis"]
                    for attr in node.attributes.values()
                )
                break

        assert found_gather_block_quantized, "No GatherBlockQuantized node found in quantized model"

    def test_rtn_quantization_with_exclusion(self, matmul_model_path, tmp_path):
        # Setup
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"bits": 4, "block_size": 128, "axis": 0, "nodes_to_exclude": ["MatMul_Node"]}
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        # Execute
        output_path = tmp_path / "excluded_quantized_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        # Assert
        assert os.path.exists(quantized_model.model_path)

        # Load the quantized model and check that no MatMulNBits nodes exist (due to exclusion)
        ir_model = ir.load(quantized_model.model_path)

        # Assert
        found_matmul_nbits = False
        found_original_matmul = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == OpType.MatMulNBits:
                found_matmul_nbits = True
            elif node.op_type == OpType.MatMul:
                found_original_matmul = True

        assert not found_matmul_nbits, "MatMulNBits node found despite exclusion"
        assert found_original_matmul, "Original MatMul node should still exist when excluded"
