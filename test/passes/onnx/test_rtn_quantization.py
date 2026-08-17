# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnxruntime import __version__ as ort_version
from packaging import version

from olive.constants import MSFT_DOMAIN, OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.rtn_quantization import OnnxBlockWiseRtnQuantization

# RTN MatMul 8-bit quantization requires onnxruntime>=1.22.0.
SKIP_8BIT_MATMUL = version.parse(ort_version) < version.parse("1.22.0")


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

    @pytest.mark.skipif(SKIP_8BIT_MATMUL, reason="RTN MatMul 8-bit quantization requires onnxruntime>=1.22.0")
    def test_rtn_quantization_preserves_graph_output_names(self, tmp_path):
        """Quantizing a MatMul that produces a graph output must not rename that output.

        External consumers (e.g. ORT GenAI's genai_config.json) reference model output
        names, so appending a `_Q{bits}` suffix to a graph output would break them.
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
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization,
            {"bits": 8, "block_size": 32, "axis": 0, "is_symmetric": False},
            disable_search=True,
            accelerator_spec=accelerator_spec,
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
        assert "hidden_Q8" in nbits_outputs, "Internal MatMul output should be renamed with the quant suffix"

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

    def test_rtn_quantization_pass_produces_valid_output_when_model_has_external_data(
        self, matmul_model_with_external_data_path, tmp_path
    ):
        """Quantizing a model with external data should produce a valid ONNX model."""
        olive_model = ONNXModelHandler(model_path=str(matmul_model_with_external_data_path))
        accelerator_spec = AcceleratorSpec(
            accelerator_type="CPU",
            execution_provider="CPUExecutionProvider",
        )
        pass_config = {"bits": 4, "block_size": 128, "axis": 0, "is_symmetric": True}
        p = create_pass_from_dict(
            OnnxBlockWiseRtnQuantization, pass_config, disable_search=True, accelerator_spec=accelerator_spec
        )

        output_path = tmp_path / "quantized_ext_data.onnx"
        quantized_model = p.run(olive_model, output_path)

        assert os.path.exists(quantized_model.model_path)

        # The output model must pass ONNX validation
        onnx.checker.check_model(quantized_model.model_path)

        ir_model = ir.load(quantized_model.model_path)
        found_matmul_nbits = False
        for node in ir_model.graph.all_nodes():
            if node.op_type == OpType.MatMulNBits:
                found_matmul_nbits = True
                break

        assert found_matmul_nbits, "No MatMulNBits node found in quantized model"

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


class TestRTNQuantizationComponentsToSkip:
    """Tests for the components_to_skip parameter on OnnxBlockWiseRtnQuantization."""

    @staticmethod
    def _make_matmul_model(tmp_path, name: str) -> ONNXModelHandler:
        """Create a tiny MatMul ONNX model and return an ONNXModelHandler."""
        weight = np.random.randn(64, 128).astype(np.float32)
        inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 64])
        out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 128])
        weight_init = onnx.helper.make_tensor(
            name="weight",
            data_type=onnx.TensorProto.FLOAT,
            dims=[64, 128],
            vals=weight.flatten().tolist(),
        )
        node = onnx.helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_Node")
        graph = onnx.helper.make_graph([node], "g", [inp], [out], initializer=[weight_init])
        model_def = onnx.helper.make_model(graph, producer_name="test")
        model_def.opset_import[0].version = 13

        model_dir = tmp_path / name
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(model_def, str(model_dir / "model.onnx"))
        return ONNXModelHandler(model_path=str(model_dir), onnx_file_name="model.onnx")

    @staticmethod
    def _make_pass(components_to_skip=None) -> OnnxBlockWiseRtnQuantization:
        accelerator_spec = AcceleratorSpec(accelerator_type="CPU", execution_provider="CPUExecutionProvider")
        config = {"bits": 4, "block_size": 128, "axis": 0, "is_symmetric": True}
        if components_to_skip is not None:
            config["components_to_skip"] = components_to_skip
        return create_pass_from_dict(
            OnnxBlockWiseRtnQuantization, config, disable_search=True, accelerator_spec=accelerator_spec
        )

    def test_components_to_skip_passes_component_through_unchanged(self, tmp_path):
        """Skipped component's model files are copied without quantization."""
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")
        embedding = self._make_matmul_model(tmp_path / "src", "embedding")

        composite = CompositeModelHandler(
            model_components=[decoder, embedding],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        p = self._make_pass(components_to_skip=["embedding"])
        result = p.run(composite, str(tmp_path / "out"))

        assert isinstance(result, CompositeModelHandler)
        assert result.model_component_names == ["decoder", "embedding"]

        # decoder should be quantized (MatMulNBits present)
        decoder_out = next(m for name, m in result.get_model_components() if name == "decoder")
        decoder_ir = ir.load(decoder_out.model_path)
        assert any(n.op_type == str(OpType.MatMulNBits) for n in decoder_ir.graph.all_nodes()), (
            "decoder should be quantized (MatMulNBits expected)"
        )

        # embedding should be unchanged (original MatMul still present)
        emb_out = next(m for name, m in result.get_model_components() if name == "embedding")
        emb_ir = ir.load(emb_out.model_path)
        has_matmul = any(n.op_type == str(OpType.MatMul) for n in emb_ir.graph.all_nodes())
        has_nbits = any(n.op_type == str(OpType.MatMulNBits) for n in emb_ir.graph.all_nodes())
        assert has_matmul, "embedding should still contain the original MatMul op"
        assert not has_nbits, "embedding should not be quantized (no MatMulNBits expected)"

    def test_components_to_skip_copies_non_default_external_data_file(self, tmp_path):
        """Skipped component using a non-default external-data filename is fully copied.

        Regression test: the copy path must discover external-data files from the
        ONNX graph itself (via ``get_external_data_file_names``) rather than assuming
        a hardcoded ``model.onnx.data`` sidecar name, since ``external_data_name`` can
        be set to any filename (e.g. ``weights.bin``).
        """
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")

        # Build an "embedding" component whose weight is stored in an external data
        # file with a non-default name ("weights.bin" instead of "model.onnx.data").
        embedding_dir = tmp_path / "src" / "embedding"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        weight = np.random.randn(64, 128).astype(np.float32)
        inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 64])
        out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 128])
        # numpy_helper.from_array (raw_data) is required here -- make_tensor(vals=...)
        # stores values inline in a way save_model's external-data conversion skips.
        weight_init = onnx.numpy_helper.from_array(weight, name="weight")
        node = onnx.helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_Node")
        graph = onnx.helper.make_graph([node], "g", [inp], [out], initializer=[weight_init])
        model_def = onnx.helper.make_model(graph, producer_name="test")
        model_def.opset_import[0].version = 13
        onnx.save_model(
            model_def,
            str(embedding_dir / "model.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
        embedding = ONNXModelHandler(model_path=str(embedding_dir), onnx_file_name="model.onnx")

        composite = CompositeModelHandler(
            model_components=[decoder, embedding],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        p = self._make_pass(components_to_skip=["embedding"])
        result = p.run(composite, str(tmp_path / "out"))

        emb_out = next(m for name, m in result.get_model_components() if name == "embedding")
        emb_out_dir = Path(emb_out.model_path).parent
        assert (emb_out_dir / "weights.bin").exists(), "non-default external-data file must be copied"

        # Loading with external data must succeed (proves the copied file is complete/valid).
        loaded = onnx.load(str(emb_out_dir / "model.onnx"), load_external_data=True)
        assert loaded.graph.initializer[0].name == "weight"

    def test_components_to_skip_none_quantizes_all(self, tmp_path):
        """When components_to_skip is not set, all composite components are quantized."""
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")
        embedding = self._make_matmul_model(tmp_path / "src", "embedding")

        composite = CompositeModelHandler(
            model_components=[decoder, embedding],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        p = self._make_pass(components_to_skip=None)
        result = p.run(composite, str(tmp_path / "out"))

        assert isinstance(result, CompositeModelHandler)

        for name, component in result.get_model_components():
            component_ir = ir.load(component.model_path)
            assert any(n.op_type == str(OpType.MatMulNBits) for n in component_ir.graph.all_nodes()), (
                f"component '{name}' should be quantized when components_to_skip is None"
            )

    def test_components_to_skip_does_not_affect_single_model(self, tmp_path):
        """components_to_skip has no effect on non-composite (single) models."""
        model = self._make_matmul_model(tmp_path, "single")
        p = self._make_pass(components_to_skip=["single"])
        result = p.run(model, str(tmp_path / "out"))

        # Single model should still be quantized despite its path matching the skip list
        result_ir = ir.load(result.model_path)
        assert any(n.op_type == str(OpType.MatMulNBits) for n in result_ir.graph.all_nodes()), (
            "Single-component model should be quantized even when components_to_skip is set"
        )

    def test_components_to_skip_in_default_config(self):
        """components_to_skip must appear in _default_config with None as default."""
        accelerator_spec = AcceleratorSpec(accelerator_type="CPU", execution_provider="CPUExecutionProvider")
        config = OnnxBlockWiseRtnQuantization._default_config(accelerator_spec)  # pylint: disable=protected-access
        assert "components_to_skip" in config
        assert config["components_to_skip"].default_value is None
        assert config["components_to_skip"].required is False

    def test_components_to_skip_unknown_name_raises(self, tmp_path):
        """Misspelled or missing component names in components_to_skip must fail loudly."""
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")
        vision = self._make_matmul_model(tmp_path / "src", "vision")
        composite = CompositeModelHandler(
            model_components=[decoder, vision],
            model_component_names=["decoder", "vision"],
        )

        p = self._make_pass(components_to_skip=["typo_component"])

        with pytest.raises(ValueError, match="typo_component") as exc_info:
            p.run(composite, str(tmp_path / "out"))

        # The error must also list the actual component names to help the user fix the typo.
        message = str(exc_info.value)
        assert "decoder" in message, message
        assert "vision" in message, message

        # Nothing should have been quantized/written before the failure.
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize(
        "malicious_name",
        ["../evil", "..", "sub/dir", "a/../../evil", os.sep + "tmp" + os.sep + "evil"],
    )
    def test_malicious_component_name_raises_before_filesystem_mutation(self, tmp_path, malicious_name):
        """Path-traversal component names must raise ValueError before touching the filesystem."""
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")
        evil = self._make_matmul_model(tmp_path / "src", "evil_src")

        # A sibling directory of the output dir that must not be deleted/overwritten.
        victim_dir = tmp_path / "evil"
        victim_dir.mkdir(parents=True, exist_ok=True)
        victim_file = victim_dir / "important.txt"
        victim_file.write_text("do not delete me")

        composite = CompositeModelHandler(
            model_components=[decoder, evil],
            model_component_names=["decoder", malicious_name],
        )

        p = self._make_pass(components_to_skip=[malicious_name])
        output_path = tmp_path / "out" / "model"

        with pytest.raises(ValueError, match="component_name must be a simple identifier"):
            p.run(composite, str(output_path))

        # No filesystem mutation may have happened: the victim file is intact and
        # not even the output directory (nor the well-named 'decoder' component) exists.
        assert victim_file.read_text() == "do not delete me"
        assert not (tmp_path / "out").exists()

    def test_malicious_component_name_raises_when_not_skipped(self, tmp_path):
        """Validation applies to all components, not only the skipped ones."""
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")
        evil = self._make_matmul_model(tmp_path / "src", "evil_src")

        composite = CompositeModelHandler(
            model_components=[decoder, evil],
            model_component_names=["decoder", "../evil"],
        )

        # Only "decoder" is skipped; the malicious name goes down the quantization path.
        p = self._make_pass(components_to_skip=["decoder"])

        with pytest.raises(ValueError, match="component_name must be a simple identifier"):
            p.run(composite, str(tmp_path / "out"))

        assert not (tmp_path / "out").exists()

    def test_components_to_skip_non_onnx_component_raises(self, tmp_path):
        """Skipping a component that is not an ONNXModelHandler must raise a clear error."""
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")
        nested_inner = self._make_matmul_model(tmp_path / "src", "inner")
        nested = CompositeModelHandler(
            model_components=[nested_inner],
            model_component_names=["inner"],
        )

        composite = CompositeModelHandler(
            model_components=[decoder, nested],
            model_component_names=["decoder", "nested"],
        )

        p = self._make_pass(components_to_skip=["nested"])

        with pytest.raises(ValueError, match="only supports ONNXModelHandler"):
            p.run(composite, str(tmp_path / "out"))

    def test_components_to_skip_copies_external_data_in_subdirectory(self, tmp_path):
        """External-data ``location`` may contain a sub-directory; the copy must handle it."""
        from olive.model.handler.composite import CompositeModelHandler

        decoder = self._make_matmul_model(tmp_path / "src", "decoder")

        embedding_dir = tmp_path / "src" / "embedding"
        (embedding_dir / "weights").mkdir(parents=True, exist_ok=True)
        weight = np.random.randn(64, 128).astype(np.float32)
        inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 64])
        out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 128])
        weight_init = onnx.numpy_helper.from_array(weight, name="weight")
        node = onnx.helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_Node")
        graph = onnx.helper.make_graph([node], "g", [inp], [out], initializer=[weight_init])
        model_def = onnx.helper.make_model(graph, producer_name="test")
        model_def.opset_import[0].version = 13
        onnx.save_model(
            model_def,
            str(embedding_dir / "model.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights/data.bin",
            size_threshold=0,
        )
        embedding = ONNXModelHandler(model_path=str(embedding_dir), onnx_file_name="model.onnx")

        composite = CompositeModelHandler(
            model_components=[decoder, embedding],
            model_component_names=["decoder", "embedding"],
            model_path=str(tmp_path / "src"),
        )

        p = self._make_pass(components_to_skip=["embedding"])
        result = p.run(composite, str(tmp_path / "out"))

        emb_out = next(m for name, m in result.get_model_components() if name == "embedding")
        emb_out_dir = Path(emb_out.model_path).parent
        assert (emb_out_dir / "weights" / "data.bin").exists(), "external-data file in a sub-directory must be copied"
        loaded = onnx.load(str(emb_out_dir / "model.onnx"), load_external_data=True)
        assert loaded.graph.initializer[0].name == "weight"
