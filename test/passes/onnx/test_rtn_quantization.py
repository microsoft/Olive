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
        # ORT GatherBlockQuantized requires quantize_axis == last dimension (data_rank - 1).
        # The gather model fixture uses 2D data [100, 64], so quantize_axis = 1.
        found_gather_block_quantized = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == str(OpType.GatherBlockQuantized):
                found_gather_block_quantized = True
                assert node.domain == MSFT_DOMAIN
                assert any(
                    attr.name == "block_size" and attr.value == pass_config["block_size"]
                    for attr in node.attributes.values()
                )
                assert any(attr.name == "quantize_axis" and attr.value == 1 for attr in node.attributes.values())
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

    @pytest.mark.parametrize("is_symmetric", [True, False])
    def test_rtn_quantization_gather_8bit(self, gather_model_path, tmp_path, is_symmetric):
        """8-bit Gather quantization should produce GatherBlockQuantized with bits=8."""
        olive_model = ONNXModelHandler(model_path=str(gather_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"bits": 8, "block_size": 128, "axis": 0, "is_symmetric": is_symmetric}
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_gather_8bit.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        ir_model = ir.load(quantized_model.model_path)

        found = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == str(OpType.GatherBlockQuantized):
                found = True
                assert node.domain == MSFT_DOMAIN
                # bits attribute must be 8
                assert any(attr.name == "bits" and attr.value == 8 for attr in node.attributes.values()), (
                    "GatherBlockQuantized should have bits=8"
                )
                # quantize_axis must be last dimension (data_rank - 1)
                assert any(attr.name == "quantize_axis" and attr.value == 1 for attr in node.attributes.values()), (
                    "quantize_axis should be forced to last dim (1 for 2-D embedding)"
                )
                break

        assert found, "No GatherBlockQuantized node found for 8-bit quantization"

    def test_rtn_quantization_gather_quantize_axis_forced_to_last_dim(self, gather_model_path, tmp_path):
        """Regardless of axis config, gather quantize_axis is forced to data_rank - 1."""
        olive_model = ONNXModelHandler(model_path=str(gather_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        # Set axis=0, but the code should force quantize_axis to last dim
        pass_config = {"bits": 4, "block_size": 128, "axis": 0, "is_symmetric": True}
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_gather_axis.onnx"
        quantized_model = p.run(olive_model, output_path)

        ir_model = ir.load(quantized_model.model_path)

        found = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == str(OpType.GatherBlockQuantized):
                found = True
                qa = [attr for attr in node.attributes.values() if attr.name == "quantize_axis"]
                assert len(qa) == 1
                assert qa[0].value == 1, f"quantize_axis should be 1 (last dim of 2-D data), got {qa[0].value}"
                break

        assert found, "No GatherBlockQuantized node found for axis/quantize_axis test"

    def test_rtn_quantization_shared_gather_weights(self, tmp_path):
        """Two Gather nodes sharing the same weight should not duplicate initializers."""
        data_shape = [100, 64]
        data_tensor = np.random.randn(*data_shape).astype(np.float32)
        data_name = "shared_data"

        data_init = onnx.helper.make_tensor(
            name=data_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=data_shape,
            vals=data_tensor.flatten().tolist(),
        )
        indices1 = onnx.helper.make_tensor_value_info("indices1", onnx.TensorProto.INT64, [1, 5])
        indices2 = onnx.helper.make_tensor_value_info("indices2", onnx.TensorProto.INT64, [1, 5])
        out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [1, 5, 64])
        out2 = onnx.helper.make_tensor_value_info("out2", onnx.TensorProto.FLOAT, [1, 5, 64])

        gather1 = onnx.helper.make_node("Gather", [data_name, "indices1"], ["out1"], name="Gather1")
        gather2 = onnx.helper.make_node("Gather", [data_name, "indices2"], ["out2"], name="Gather2")

        graph = onnx.helper.make_graph(
            [gather1, gather2],
            "shared_weight_test",
            [indices1, indices2],
            [out1, out2],
            initializer=[data_init],
        )
        model = onnx.helper.make_model(graph, producer_name="olive-test")
        model.opset_import[0].version = 13
        model.ir_version = 10

        model_path = tmp_path / "shared_gather.onnx"
        onnx.save(model, str(model_path))

        olive_model = ONNXModelHandler(model_path=str(model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization,
            {"bits": 4, "block_size": 128, "axis": 0, "is_symmetric": True},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        output_path = tmp_path / "shared_gather_quantized.onnx"
        quantized_model = p.run(olive_model, output_path)

        ir_model = ir.load(quantized_model.model_path)

        # Both nodes should be GatherBlockQuantized
        gbq_nodes = [n for n in ir_model.graph.all_nodes() if n.op_type == str(OpType.GatherBlockQuantized)]
        assert len(gbq_nodes) == 2, f"Expected 2 GatherBlockQuantized nodes, got {len(gbq_nodes)}"

        # The quantized data inputs (first input) should refer to the same name
        quant_data_names = [n.inputs[0].name for n in gbq_nodes]
        assert quant_data_names[0] == quant_data_names[1], (
            f"Shared weight should produce same quantized initializer name: {quant_data_names}"
        )

    def test_rtn_quantization_removes_unused_initializers(self, matmul_model_path, tmp_path):
        """After quantization, original FP32 weight initializers should be removed."""
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization,
            {"bits": 4, "block_size": 128, "axis": 0, "is_symmetric": True},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        output_path = tmp_path / "unused_init_test.onnx"
        quantized_model = p.run(olive_model, output_path)

        ir_model = ir.load(quantized_model.model_path)

        # The original "weight" initializer should be gone
        init_names = set(ir_model.graph.initializers.keys())
        assert "weight" not in init_names, (
            f"Original FP32 'weight' initializer should have been removed, found: {init_names}"
        )
