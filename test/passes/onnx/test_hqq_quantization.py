# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

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

    @pytest.fixture
    def matmul_model_with_external_data_path(self, tmp_path):
        """Create an ONNX model with weights stored as external data."""
        input_shape = [1, 64]
        weight_shape = [64, 128]
        weight_tensor = np.random.randn(*weight_shape).astype(np.float32)

        input_name = "input"
        output_name = "output"
        weight_name = "weight"

        input_tensor_proto = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        output_tensor_proto = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 128])
        weight_tensor_proto = onnx.numpy_helper.from_array(weight_tensor, name=weight_name)

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

        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13

        model_path = str(tmp_path / "matmul_model_ext.onnx")
        onnx.save(
            model_def,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="matmul_model_ext.onnx.data",
            size_threshold=0,
        )
        return tmp_path / "matmul_model_ext.onnx"

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

    @pytest.mark.parametrize(
        ("config_key", "pattern", "expect_quantized"),
        [
            ("nodes_to_exclude", "MatMul_*", False),
            ("nodes_to_include", "MatMul_*", True),
            ("nodes_to_include", "Other_*", False),
        ],
    )
    def test_hqq_quantization_with_node_glob(self, matmul_model_path, tmp_path, config_key, pattern, expect_quantized):
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"block_size": 128, config_key: [pattern]}
        quantization_pass = create_pass_from_dict(
            OnnxHqqQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        quantized_model = quantization_pass.run(olive_model, tmp_path / "quantized_glob_model.onnx")
        ir_model = ir.load(quantized_model.model_path)
        found_matmul_nbits = any(node.op_type == OpType.MatMulNBits for node in ir_model.graph.all_nodes())

        assert found_matmul_nbits is expect_quantized

    @pytest.mark.parametrize("input_size", [96, 100])
    def test_hqq_quantization_preserves_matmul_nbits_layout(self, monkeypatch, tmp_path, input_size):
        """The emitted MatMulNBits initializers must preserve HQQ's N-major representation."""
        # pylint: disable=protected-access
        block_size = 32
        output_size = 40
        rng = np.random.default_rng(0)
        weight = rng.normal(scale=0.02, size=(input_size, output_size)).astype(np.float32)
        weight[0, 0] = 100
        weight[-1, -1] = -100

        graph = onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_Node")],
            name="hqq-layout",
            inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, input_size])],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, output_size])],
            initializer=[onnx.numpy_helper.from_array(weight, name="weight")],
        )
        model = onnx.helper.make_model(graph, producer_name="olive-test")
        model.opset_import[0].version = 13
        model_path = tmp_path / "matmul.onnx"
        onnx.save(model, model_path)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        quantization_pass = create_pass_from_dict(
            OnnxHqqQuantization,
            {"block_size": block_size},
            disable_search=True,
            accelerator_spec=AcceleratorSpec(
                accelerator_type="CPU",
                execution_provider="CPUExecutionProvider",
            ),
        )
        expected_quantized, expected_scales, expected_zero_points = quantization_pass._quantize_internal(
            torch.from_numpy(weight.T.copy()), group_size=block_size, axis=1
        )
        quantized_model = quantization_pass.run(
            ONNXModelHandler(model_path=str(model_path)),
            tmp_path / "quantized.onnx",
        )
        ir_model = ir.load(quantized_model.model_path)
        matmul_nbits = next(node for node in ir_model.graph.all_nodes() if node.op_type == OpType.MatMulNBits)

        packed = matmul_nbits.inputs[1].const_value.numpy()
        scales = matmul_nbits.inputs[2].const_value.numpy()
        zero_points = matmul_nbits.inputs[3].const_value.numpy()

        k_blocks = (input_size + block_size - 1) // block_size
        padded_input_size = k_blocks * block_size
        expected_shape = (output_size, padded_input_size)
        assert packed.shape == (output_size, k_blocks, block_size // 2)
        assert matmul_nbits.attributes.get_int("K") == input_size
        assert matmul_nbits.attributes.get_int("N") == output_size
        packed = packed.reshape(output_size, -1)
        unpacked = np.empty(expected_shape, dtype=np.int32)
        unpacked[:, 0::2] = packed & 0x0F
        unpacked[:, 1::2] = packed >> 4

        np.testing.assert_array_equal(unpacked, expected_quantized.numpy())
        np.testing.assert_array_equal(scales.reshape(output_size, k_blocks), expected_scales.numpy())
        np.testing.assert_array_equal(
            zero_points.reshape(output_size, k_blocks),
            expected_zero_points.numpy(),
        )

    def test_hqq_quantization_preserves_graph_output_names(self, tmp_path):
        """Quantizing a MatMul that produces a graph output must not rename that output.

        External consumers (e.g. ORT GenAI's genai_config.json) reference model output
        names, so appending a `_Q4` suffix to a graph output would break them. Internal
        tensors, however, are still renamed.
        """
        # X[2,64] -> MatMul(W1) -> hidden (internal) -> MatMul(W2) -> audio_features (graph output)
        w1 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W1")
        w2 = onnx.numpy_helper.from_array(np.random.randn(128, 64).astype(np.float32), name="W2")
        internal_matmul = onnx.helper.make_node("MatMul", ["X", "W1"], ["hidden"], name="enc/MatMul")
        terminal_matmul = onnx.helper.make_node("MatMul", ["hidden", "W2"], ["audio_features"], name="projector/MatMul")

        graph_def = onnx.helper.make_graph(
            nodes=[internal_matmul, terminal_matmul],
            name="graph-output-test",
            inputs=[onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 64])],
            outputs=[onnx.helper.make_tensor_value_info("audio_features", onnx.TensorProto.FLOAT, [2, 64])],
            initializer=[w1, w2],
        )
        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13
        model_def.ir_version = 10
        model_path = tmp_path / "graph_output_model.onnx"
        onnx.save(model_def, str(model_path))

        olive_model = ONNXModelHandler(model_path=str(model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        p = create_pass_from_dict(
            OnnxHqqQuantization, {"block_size": 128}, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "graph_output_quantized.onnx"
        quantized_model = p.run(olive_model, output_path)

        ir_model = ir.load(quantized_model.model_path)

        # The graph output name must be preserved exactly.
        output_names = [o.name for o in ir_model.graph.outputs]
        assert output_names == ["audio_features"], f"Graph output was renamed: {output_names}"

        # Both MatMuls should be quantized, and the internal tensor should still be renamed.
        nbits_outputs = [
            o.name for n in ir_model.graph.all_nodes() if n.op_type == OpType.MatMulNBits for o in n.outputs
        ]
        assert "audio_features" in nbits_outputs, "Terminal MatMul should keep the graph output name"
        assert "hidden_Q4" in nbits_outputs, "Internal MatMul output should be renamed with the quant suffix"

    def test_hqq_quantization_removes_replaced_weight_initializers(self, tmp_path):
        """Weights superseded by MatMulNBits must be deleted, not left as orphaned initializers.

        Olive's K-Quant and RTN passes prune unused initializers after quantization. Without
        the same step the original FP16/FP32 weight stays registered in the graph even though
        no node consumes it, inflating the saved package by the full size of the original
        weights while changing nothing numerically.
        """
        w1 = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W1")
        w2 = onnx.numpy_helper.from_array(np.random.randn(128, 64).astype(np.float32), name="W2")
        quantized_matmul = onnx.helper.make_node("MatMul", ["X", "W1"], ["hidden"], name="enc/MatMul")
        excluded_matmul = onnx.helper.make_node("MatMul", ["hidden", "W2"], ["output"], name="skip/MatMul")

        graph_def = onnx.helper.make_graph(
            nodes=[quantized_matmul, excluded_matmul],
            name="orphan-initializer-test",
            inputs=[onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 64])],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 64])],
            initializer=[w1, w2],
        )
        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13
        model_def.ir_version = 10
        model_path = tmp_path / "orphan_model.onnx"
        onnx.save(model_def, str(model_path))

        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        p = create_pass_from_dict(
            OnnxHqqQuantization,
            {"block_size": 32, "nodes_to_exclude": ["skip/MatMul"]},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        quantized_model = p.run(ONNXModelHandler(model_path=str(model_path)), tmp_path / "orphan_quantized.onnx")
        ir_model = ir.load(quantized_model.model_path)

        initializer_names = set(ir_model.graph.initializers)
        assert "W1" not in initializer_names, "Replaced weight was left behind as an orphaned initializer"
        assert "W2" in initializer_names, "Weight of an excluded MatMul must be preserved"

        used_names = {inp.name for node in ir_model.graph.all_nodes() for inp in node.inputs if inp is not None}
        assert not (initializer_names - used_names), "Quantized model still contains orphaned initializers"

    def test_hqq_quantization_removes_replaced_initializer_graph_input(self, tmp_path):
        weight = onnx.numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="W")
        graph_def = onnx.helper.make_graph(
            nodes=[onnx.helper.make_node("MatMul", ["X", "W"], ["output"], name="MatMul")],
            name="initializer-graph-input-test",
            inputs=[
                onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 64]),
                onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [64, 128]),
            ],
            outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 128])],
            initializer=[weight],
        )
        model_def = onnx.helper.make_model(graph_def, producer_name="olive-test")
        model_def.opset_import[0].version = 13
        model_def.ir_version = 10
        model_path = tmp_path / "initializer_input.onnx"
        onnx.save(model_def, model_path)

        p = create_pass_from_dict(
            OnnxHqqQuantization,
            {"block_size": 32},
            disable_search=True,
            accelerator_spec=AcceleratorSpec(
                accelerator_type="CPU",
                execution_provider="CPUExecutionProvider",
            ),
        )
        quantized_model = p.run(
            ONNXModelHandler(model_path=str(model_path)),
            tmp_path / "initializer_input_quantized.onnx",
        )

        model_proto = onnx.load(quantized_model.model_path, load_external_data=False)
        assert "W" not in {value.name for value in model_proto.graph.initializer}
        assert "W" not in {value.name for value in model_proto.graph.input}

        session = ort.InferenceSession(
            quantized_model.model_path,
            providers=["CPUExecutionProvider"],
        )
        output = session.run(None, {"X": np.ones((1, 64), dtype=np.float32)})
        assert output[0].shape == (1, 128)

    def test_hqq_quantization_pass_produces_valid_output_when_model_has_external_data(
        self, matmul_model_with_external_data_path, tmp_path
    ):
        """Quantizing a model with external data should produce a valid ONNX model."""
        olive_model = ONNXModelHandler(model_path=str(matmul_model_with_external_data_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"block_size": 128}
        p = create_pass_from_dict(
            OnnxHqqQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_ext_data.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        # The output model must pass ONNX validation (regression test for #2223)
        onnx.checker.check_model(quantized_model.model_path)

        ir_model = ir.load(quantized_model.model_path)
        found_matmul_nbits = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == OpType.MatMulNBits:
                found_matmul_nbits = True
                break

        assert found_matmul_nbits, "No MatMulNBits node found in quantized model"
