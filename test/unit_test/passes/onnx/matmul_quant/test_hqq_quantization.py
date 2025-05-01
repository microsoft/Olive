# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
import torch

from olive.model import ONNXModelHandler
from olive.passes.onnx.hqq_quantization import HqqQuantizer, OnnxHqqQuantization
from olive.passes.onnx.matmul_quant.utils import MSFT_DOMAIN, OpType


class TestHQQQuantization:
    @pytest.fixture
    def matmul_model_path(self, tmp_path):
        """Create a simple ONNX model with a MatMul op and save it to a temporary file."""
        # Create input tensor
        input_shape = [1, 64]
        weight_shape = [64, 128]
        input_tensor = np.random.randn(*input_shape).astype(np.float32)
        weight_tensor = np.random.randn(*weight_shape).astype(np.float32)

        # Create model
        input_name = "input"
        output_name = "output"
        weight_name = "weight"

        input_tensor_proto = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        output_tensor_proto = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 128])
        weight_tensor_proto = onnx.helper.make_tensor(
            name=weight_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=weight_shape,
            vals=weight_tensor.flatten().tolist()
        )

        # Create MatMul node
        node_def = onnx.helper.make_node(
            "MatMul",
            inputs=[input_name, weight_name],
            outputs=[output_name],
            name="MatMul_Node"
        )

        # Create graph
        graph_def = onnx.helper.make_graph(
            nodes=[node_def],
            name="test-model",
            inputs=[input_tensor_proto],
            outputs=[output_tensor_proto],
            initializer=[weight_tensor_proto]
        )

        # Create model
        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13

        # Save model
        model_path = tmp_path / "matmul_model.onnx"
        onnx.save(model_def, str(model_path))
        return model_path

    def test_process_subgraph(self, matmul_model_path):
        """Test that _process_with_dag correctly quantizes MatMul nodes and updates the model."""
        # Load model
        model = ONNXModelHandler(model_path=str(matmul_model_path))
        
        # Create pass with default config
        pass_config = {
            "block_size": 128,
            "axis": 1,
            "nodes_to_exclude": None,
            "nodes_to_include": None
        }
        hqq_pass = OnnxHqqQuantization()
        
        # Set config directly to bypass config validation
        hqq_pass.config = MagicMock()
        hqq_pass.config.block_size = pass_config["block_size"]
        hqq_pass.config.axis = pass_config["axis"]
        hqq_pass.config.nodes_to_exclude = pass_config["nodes_to_exclude"]
        hqq_pass.config.nodes_to_include = pass_config["nodes_to_include"]
        
        # 加载模型
        loaded_model = model.load_model()
        hqq_pass.model = loaded_model
        
        # 创建DAG并处理
        from olive.passes.onnx.onnx_dag import OnnxDAG
        dag = OnnxDAG(loaded_model)
        hqq_pass._process_with_dag(dag)
        
        # 更新模型
        dag.update()
        
        # Check that at least one node in the processed graph is of type MatMulNBits
        found_matmul_nbits = False
        for node in loaded_model.graph.node:
            if node.op_type == str(OpType.MatMulNBits):
                found_matmul_nbits = True
                # Verify that the node has the expected domain and attributes
                assert node.domain == MSFT_DOMAIN
                assert any(attr.name == "bits" and attr.i == 4 for attr in node.attribute)
                assert any(attr.name == "block_size" and attr.i == pass_config["block_size"] for attr in node.attribute)
                break
        
        assert found_matmul_nbits, "No MatMulNBits node found in processed graph"

    def test_run_for_config(self, matmul_model_path, tmp_path):
        """Test the full execution of the HQQ quantization pass."""
        # Load model
        model = ONNXModelHandler(model_path=str(matmul_model_path))
        
        # Create pass config
        pass_config = MagicMock()
        pass_config.block_size = 128
        pass_config.axis = 1
        pass_config.nodes_to_exclude = None
        pass_config.nodes_to_include = None
        
        # Set up output path
        output_path = tmp_path / "quantized_model.onnx"
        
        # Run the pass
        hqq_pass = OnnxHqqQuantization()
        quantized_model = hqq_pass._run_for_config(model, pass_config, str(output_path))
        
        # Check that the quantized model exists
        assert os.path.exists(quantized_model.model_path)
        
        # Load the quantized model and check for MatMulNBits nodes
        quantized_onnx = onnx.load(quantized_model.model_path)
        
        found_matmul_nbits = False
        for node in quantized_onnx.graph.node:
            if node.op_type == "MatMulNBits":
                found_matmul_nbits = True
                break
        
        assert found_matmul_nbits, "No MatMulNBits node found in quantized model"

    def test_quantizer_integration(self, matmul_model_path, tmp_path):
        """Test the HQQQuantizer class integration with the pass."""
        # Create a HQQQuantizer instance
        quantizer = HqqQuantizer(block_size=128, axis=1)
        
        # Load model
        model = ONNXModelHandler(model_path=str(matmul_model_path))
        loaded_model = model.load_model()
        
        # Find the MatMul node
        matmul_node = None
        for node in loaded_model.graph.node:
            if node.op_type == "MatMul":
                matmul_node = node
                break
        
        assert matmul_node is not None, "MatMul node not found in test model"
        
        # Quantize the node
        graph_stack = [loaded_model.graph]
        quantized_nodes = quantizer.quantize(matmul_node, graph_stack)
        
        # Check that the node was quantized
        assert len(quantized_nodes) == 1
        assert quantized_nodes[0].op_type == str(OpType.MatMulNBits)
        assert quantized_nodes[0].domain == MSFT_DOMAIN
        
    def test_pack_on_row_fast_248bit_with_4bit(self):
        """Test that pack_on_row_fast_248bit works correctly with 4-bit quantization."""
        # Create tensors with known shapes
        # For 4-bit packing, we need 2 values per byte
        ori_int_tensor = torch.randint(0, 16, (128, 64), dtype=torch.int)  # 原始整数张量
        pack_tensor = torch.zeros((128, 32), dtype=torch.uint8)  # 目标打包张量 (64/2 = 32)
        
        # Execute packing
        HqqQuantizer.pack_on_row_fast_248bit(pack_tensor, ori_int_tensor)
        
        # Sample and verify a few values
        # For 4-bit packing, each byte should contain 2 values
        for _ in range(5):  # Test 5 random positions
            row = np.random.randint(0, 128)
            col = np.random.randint(0, 32)
            byte_val = pack_tensor[row, col].item()
            
            # 解包这个字节，检查两个4位值是否正确
            orig_col = col * 2
            val0 = byte_val & 0x0F
            val1 = (byte_val >> 4) & 0x0F
            
            # 检查解包的值是否与原始值匹配
            assert val0 == ori_int_tensor[row, orig_col].item()
            if orig_col + 1 < ori_int_tensor.shape[1]:
                assert val1 == ori_int_tensor[row, orig_col + 1].item() 