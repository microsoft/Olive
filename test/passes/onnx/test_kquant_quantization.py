# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import numpy as np
import onnx
import pytest
from onnxruntime import __version__ as ort_version
from packaging import version

from olive.constants import OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.kquant_quantization import OnnxKQuantQuantization

# 8-bit MatMul quantization requires onnxruntime>=1.22.0.
SKIP_8BIT_MATMUL = version.parse(ort_version) < version.parse("1.22.0")


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

    def test_kquant_with_nodes_to_exclude_glob(self, matmul_model_path, tmp_path):
        """Test k-quant where nodes_to_exclude uses a glob pattern."""
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        # "*_1" matches MatMul_1 only; MatMul_2 should still be quantized.
        pass_config = {
            "bits": 4,
            "block_size": 32,
            "nodes_to_exclude": ["*_1"],
        }
        p = create_pass_from_dict(
            OnnxKQuantQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_glob_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        quantized_onnx = onnx.load(quantized_model.model_path)
        matmul_nbits_nodes = [n for n in quantized_onnx.graph.node if n.op_type == str(OpType.MatMulNBits)]
        matmul_nodes = [n for n in quantized_onnx.graph.node if n.op_type == "MatMul"]

        assert len(matmul_nbits_nodes) == 1, "Expected 1 MatMulNBits node (MatMul_2 quantized)"
        assert len(matmul_nodes) == 1, "Expected 1 original MatMul node (MatMul_1 excluded via glob)"

    def test_kquant_exclude_entry_with_metachars_is_exact(self, matmul_model_path, tmp_path):
        """An exclusion entry with fnmatch metacharacters but no wildcard is matched exactly.

        'MatMul_[12]' must NOT glob-match node 'MatMul_1'; without a '*'/'?' wildcard the entry is
        treated as an exact name, so both MatMuls are still quantized.
        """
        olive_model = ONNXModelHandler(model_path=str(matmul_model_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {
            "bits": 4,
            "block_size": 32,
            "nodes_to_exclude": ["MatMul_[12]"],
        }
        p = create_pass_from_dict(
            OnnxKQuantQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_meta_model.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        quantized_onnx = onnx.load(quantized_model.model_path)
        matmul_nbits_nodes = [n for n in quantized_onnx.graph.node if n.op_type == str(OpType.MatMulNBits)]

        assert len(matmul_nbits_nodes) == 2, "Expected both MatMuls quantized (bracket entry not glob-matched)"

    @pytest.mark.skipif(SKIP_8BIT_MATMUL, reason="8-bit MatMul quantization requires onnxruntime>=1.22.0")
    def test_kquant_preserves_graph_output_names(self, tmp_path):
        """Quantizing a MatMul that produces a graph output must not rename that output.

        External consumers (e.g. ORT GenAI's genai_config.json) reference model output
        names, so appending a quant suffix to a graph output would break them.
        Internal tensors, however, are still renamed.
        """
        # X[2,4] -> MatMul(W1) -> hidden (internal) -> MatMul(W2) -> audio_features (graph output)
        w1 = onnx.numpy_helper.from_array(np.random.randn(4, 5).astype(np.float32), name="W1")
        w2 = onnx.numpy_helper.from_array(np.random.randn(5, 3).astype(np.float32), name="W2")
        internal_matmul = onnx.helper.make_node("MatMul", ["X", "W1"], ["hidden"], name="enc/MatMul")
        terminal_matmul = onnx.helper.make_node("MatMul", ["hidden", "W2"], ["audio_features"], name="projector/MatMul")

        graph_def = onnx.helper.make_graph(
            nodes=[internal_matmul, terminal_matmul],
            name="graph-output-test",
            inputs=[onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])],
            outputs=[onnx.helper.make_tensor_value_info("audio_features", onnx.TensorProto.FLOAT, [2, 3])],
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
        pass_config = {"bits": 8, "block_size": 32}
        p = create_pass_from_dict(
            OnnxKQuantQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "graph_output_quantized.onnx"
        quantized_model = p.run(olive_model, output_path)

        quantized_onnx = onnx.load(quantized_model.model_path)

        # The graph output name must be preserved exactly.
        output_names = [o.name for o in quantized_onnx.graph.output]
        assert output_names == ["audio_features"], f"Graph output was renamed: {output_names}"

        # Both MatMuls should be quantized, and the internal tensor should still be renamed.
        nbits_outputs = [o for n in quantized_onnx.graph.node if n.op_type == str(OpType.MatMulNBits) for o in n.output]
        assert "audio_features" in nbits_outputs, "Terminal MatMul should keep the graph output name"
        assert "hidden_Q8" in nbits_outputs, "Internal MatMul output should be renamed with the quant suffix"
