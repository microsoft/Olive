# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.onnx.peephole_optimizer import ModelOptimizer, OnnxPeepholeOptimizer
from test.utils import get_onnx_model


@pytest.fixture(name="external_data_config")
def external_data_config_fixture():
    return {
        "save_as_external_data": False,
        "all_tensors_to_one_file": True,
        "external_data_name": None,
        "size_threshold": 1024,
        "convert_attribute": False,
    }


def test_onnx_peephole_optimizer_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxPeepholeOptimizer, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)


# TODO(team): this test will creat an unnecessary intermediate model file. Need to optimize it.
def test_onnx_peephole_optimizer_pass_fuse_reshape_operations(tmp_path, external_data_config):
    X = helper.make_tensor_value_info("X", TensorProto.INT64, [None])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])  # noqa: N806

    node_def_1 = helper.make_node("Reshape", ["X", "shape_1"], ["Z"], domain="com.microsoft")
    node_def_2 = helper.make_node("Reshape", ["Z", "shape_2"], ["Y"], domain="com.microsoft")

    shape_data_1 = np.array([2, 4, 6], np.int64)
    shape_init_1 = helper.make_tensor(
        name="shape_1",
        data_type=TensorProto.INT64,
        dims=shape_data_1.shape,
        vals=shape_data_1.tobytes(),
        raw=True,
    )

    shape_data_2 = np.array([-1], np.int64)
    shape_init_2 = helper.make_tensor(
        name="shape_2",
        data_type=TensorProto.INT64,
        dims=shape_data_2.shape,
        vals=shape_data_2.tobytes(),
        raw=True,
    )

    graph_def = helper.make_graph(
        [node_def_1, node_def_2],
        "g",
        [X],
        [Y],
        initializer=[shape_init_1, shape_init_2],
    )
    opset_imports = [
        helper.make_operatorsetid("", 18),
        helper.make_operatorsetid("com.microsoft", 1),
    ]
    model = helper.make_model(
        graph_def,
        producer_name="From test_peephole_optimizer.py",
        opset_imports=opset_imports,
    )

    m = model_proto_to_olive_model(model, str(tmp_path / "input.onnx"), external_data_config)
    p = create_pass_from_dict(
        OnnxPeepholeOptimizer, external_data_config, disable_search=True, accelerator_spec=DEFAULT_CPU_ACCELERATOR
    )

    actual_model = p.run(m, str(tmp_path / "onnx"))
    assert Path(actual_model.model_path).exists()

    actual_model = actual_model.load_model()
    assert len(actual_model.graph.node) == 1

    reshape_op_count = 0
    others_op_count = 0
    for node in actual_model.graph.node:
        if node.op_type == "Reshape":
            reshape_op_count += 1
        else:
            others_op_count += 1

    assert reshape_op_count == 1
    assert others_op_count == 0


@patch("olive.passes.onnx.peephole_optimizer.model_proto_to_olive_model")
@patch("onnxoptimizer.optimize")
@patch("onnxscript.optimizer.optimize")
def test_onnxscript(mock_onnxscript, mock_onnxoptimizer, mock_model_proto_to_olive_model, tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxPeepholeOptimizer, {}, disable_search=True)
    mock_onnxscript.return_value = input_model.load_model()
    mock_onnxoptimizer.return_value = input_model.load_model()
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)

    # assert
    mock_onnxscript.assert_called_once_with(input_model.load_model())


@patch("olive.passes.onnx.peephole_optimizer.model_proto_to_olive_model")
@patch("onnxoptimizer.optimize")
@patch("onnxscript.optimizer.optimize")
def test_onnxoptimizer(mock_onnxscript, mock_onnxoptimizer, mock_model_proto_to_olive_model, tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxPeepholeOptimizer, {}, disable_search=True)
    mock_onnxscript.return_value = input_model.load_model()
    mock_onnxoptimizer.return_value = input_model.load_model()
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)

    # assert
    mock_onnxoptimizer.assert_called_once()


# ── _ensure_com_microsoft_opset ────────────────────────────────────────────


class TestEnsureComMicrosoftOpset:
    """Unit tests for ModelOptimizer.ensure_com_microsoft_opset."""

    def _make_optimizer_with_model(self, model, tmp_path):
        """Save a model to disk and create a ModelOptimizer around it."""
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        opt = ModelOptimizer(str(path))
        opt.model = model  # use the in-memory model directly
        return opt

    def test_adds_opset_when_missing(self, tmp_path):
        model = helper.make_model(
            helper.make_graph([], "g", [], []),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        assert not any(op.domain == "com.microsoft" for op in model.opset_import)

        opt = self._make_optimizer_with_model(model, tmp_path)
        opt.ensure_com_microsoft_opset()

        assert any(op.domain == "com.microsoft" for op in opt.model.opset_import)

    def test_does_not_duplicate_opset(self, tmp_path):
        model = helper.make_model(
            helper.make_graph([], "g", [], []),
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        )

        opt = self._make_optimizer_with_model(model, tmp_path)
        opt.ensure_com_microsoft_opset()

        ms_opsets = [op for op in opt.model.opset_import if op.domain == "com.microsoft"]
        assert len(ms_opsets) == 1

    def test_adds_opset_to_functions(self, tmp_path):
        func = onnx.FunctionProto()
        func.name = "test_func"
        func.domain = "test.domain"
        func.opset_import.append(helper.make_opsetid("", 17))

        model = helper.make_model(
            helper.make_graph([], "g", [], []),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.functions.append(func)

        opt = self._make_optimizer_with_model(model, tmp_path)
        opt.ensure_com_microsoft_opset()

        func_domains = {op.domain for op in opt.model.functions[0].opset_import}
        assert "com.microsoft" in func_domains

    def test_skips_function_with_existing_opset(self, tmp_path):
        func = onnx.FunctionProto()
        func.name = "test_func"
        func.domain = "test.domain"
        func.opset_import.append(helper.make_opsetid("", 17))
        func.opset_import.append(helper.make_opsetid("com.microsoft", 1))

        model = helper.make_model(
            helper.make_graph([], "g", [], []),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.functions.append(func)

        opt = self._make_optimizer_with_model(model, tmp_path)
        opt.ensure_com_microsoft_opset()

        ms_opsets = [op for op in opt.model.functions[0].opset_import if op.domain == "com.microsoft"]
        assert len(ms_opsets) == 1


# ── Cast chain elimination via OnnxPeepholeOptimizer ───────────────────────


class TestCastChainElimination:
    """Tests for cast chain elimination integrated into OnnxPeepholeOptimizer."""

    @pytest.fixture
    def cast_chain_model_path(self, tmp_path):
        """Model with a redundant Cast chain: fp32 → fp16 → fp32."""
        input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

        cast_to_fp16 = helper.make_node("Cast", ["X"], ["x_fp16"], to=TensorProto.FLOAT16)
        cast_back = helper.make_node("Cast", ["x_fp16"], ["Y"], to=TensorProto.FLOAT)

        graph = helper.make_graph([cast_to_fp16, cast_back], "cast_chain", [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10  # Compatible with ORT version in CI
        onnx.checker.check_model(model)

        path = tmp_path / "cast_chain.onnx"
        onnx.save(model, str(path))
        return path

    def test_eliminates_redundant_cast_chain(self, cast_chain_model_path, tmp_path):
        olive_model = ONNXModelHandler(model_path=str(cast_chain_model_path))
        p = create_pass_from_dict(OnnxPeepholeOptimizer, {"cast_chain_elimination": True}, disable_search=True)
        output = p.run(olive_model, str(tmp_path / "out.onnx"))

        result_model = onnx.load(output.model_path)
        # The rewrite rule collapses the round-trip fp32→fp16→fp32 chain
        # into a single Identity node.
        assert len(result_model.graph.node) == 1
        assert result_model.graph.node[0].op_type == "Identity"

    def test_opset_fixup_applied(self, cast_chain_model_path, tmp_path):
        olive_model = ONNXModelHandler(model_path=str(cast_chain_model_path))
        p = create_pass_from_dict(
            OnnxPeepholeOptimizer,
            {"fix_com_microsoft_opset": True, "cast_chain_elimination": False},
            disable_search=True,
        )
        output = p.run(olive_model, str(tmp_path / "out.onnx"))

        result_model = onnx.load(output.model_path)
        assert any(op.domain == "com.microsoft" for op in result_model.opset_import)

    def test_numerical_correctness(self, cast_chain_model_path, tmp_path):
        """Optimized model should produce the same output as the original."""
        olive_model = ONNXModelHandler(model_path=str(cast_chain_model_path))
        p = create_pass_from_dict(OnnxPeepholeOptimizer, {"cast_chain_elimination": True}, disable_search=True)
        output = p.run(olive_model, str(tmp_path / "out.onnx"))

        x = np.random.randn(2, 4).astype(np.float32)

        orig_sess = ort.InferenceSession(str(cast_chain_model_path), providers=["CPUExecutionProvider"])
        opt_sess = ort.InferenceSession(output.model_path, providers=["CPUExecutionProvider"])

        orig_out = orig_sess.run(None, {"X": x})[0]
        opt_out = opt_sess.run(None, {"X": x})[0]
        np.testing.assert_allclose(orig_out, opt_out, atol=1e-3)
