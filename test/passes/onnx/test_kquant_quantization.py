# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import numpy as np
import onnx
import pytest

from olive.constants import OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.kquant_quantization import OnnxKQuantQuantization


class TestKQuantQuantization:
    @pytest.fixture
    def matmul_model_path(self, tmp_path):
        """Create a simple ONNX model with two MatMul ops."""
        input_shape = [1, 64]
        weight_shape = [64, 128]
        weight1 = np.random.randn(*weight_shape).astype(np.float32)
        weight2 = np.random.randn(128, 64).astype(np.float32)

        input_name = "input"
        mid_name = "mid"
        output_name = "output"

        input_proto = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        output_proto = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 64])

        weight1_proto = onnx.helper.make_tensor(
            name="weight1",
            data_type=onnx.TensorProto.FLOAT,
            dims=weight_shape,
            vals=weight1.flatten().tolist(),
        )
        weight2_proto = onnx.helper.make_tensor(
            name="weight2",
            data_type=onnx.TensorProto.FLOAT,
            dims=[128, 64],
            vals=weight2.flatten().tolist(),
        )

        matmul1 = onnx.helper.make_node("MatMul", inputs=[input_name, "weight1"], outputs=[mid_name], name="MatMul_1")
        matmul2 = onnx.helper.make_node("MatMul", inputs=[mid_name, "weight2"], outputs=[output_name], name="MatMul_2")

        graph_def = onnx.helper.make_graph(
            nodes=[matmul1, matmul2],
            name="test-model",
            inputs=[input_proto],
            outputs=[output_proto],
            initializer=[weight1_proto, weight2_proto],
        )

        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13

        model_path = tmp_path / "matmul_model.onnx"
        onnx.save(model_def, str(model_path))
        return model_path

    def test_kquant_basic(self, matmul_model_path, tmp_path):
        """Test basic k-quant quantization (uniform INT4)."""
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"bits": 4, "block_size": 32}
        p = create_pass_from_dict(
            OnnxKQuantQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        quantized_onnx = onnx.load(quantized_model.model_path)
        matmul_nbits_nodes = [n for n in quantized_onnx.graph.node if n.op_type == str(OpType.MatMulNBits)]
        assert len(matmul_nbits_nodes) == 2, "Expected 2 MatMulNBits nodes for uniform k-quant"

    def test_kquant_with_customized_weight_config(self, matmul_model_path, tmp_path):
        """Test k-quant with per-node config overrides (different group_size)."""
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {
            "bits": 4,
            "block_size": 32,
            "customized_weight_config": {"MatMul_1": {"bits": 4, "group_size": 64}},
        }
        p = create_pass_from_dict(
            OnnxKQuantQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_mixed_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        quantized_onnx = onnx.load(quantized_model.model_path)
        matmul_nbits_nodes = [n for n in quantized_onnx.graph.node if n.op_type == str(OpType.MatMulNBits)]
        assert len(matmul_nbits_nodes) == 2, "Expected 2 MatMulNBits nodes for k-quant with overrides"

    def test_kquant_with_nodes_to_exclude(self, matmul_model_path, tmp_path):
        """Test k-quant with node exclusion."""
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {
            "bits": 4,
            "block_size": 32,
            "nodes_to_exclude": ["MatMul_1"],
        }
        p = create_pass_from_dict(
            OnnxKQuantQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_excluded_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        quantized_onnx = onnx.load(quantized_model.model_path)
        matmul_nbits_nodes = [n for n in quantized_onnx.graph.node if n.op_type == str(OpType.MatMulNBits)]
        matmul_nodes = [n for n in quantized_onnx.graph.node if n.op_type == "MatMul"]

        assert len(matmul_nbits_nodes) == 1, "Expected 1 MatMulNBits node (MatMul_2 quantized)"
        assert len(matmul_nodes) == 1, "Expected 1 original MatMul node (MatMul_1 excluded)"
