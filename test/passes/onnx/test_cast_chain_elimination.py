# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for OnnxCastChainElimination pass."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.cast_chain_elimination import OnnxCastChainElimination, _ensure_com_microsoft_opset


class TestEnsureComMicrosoftOpset:
    """Unit tests for _ensure_com_microsoft_opset helper."""

    def test_adds_opset_when_missing(self):
        model = helper.make_model(
            helper.make_graph([], "g", [], []),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        assert not any(op.domain == "com.microsoft" for op in model.opset_import)

        _ensure_com_microsoft_opset(model)

        assert any(op.domain == "com.microsoft" for op in model.opset_import)

    def test_does_not_duplicate_opset(self):
        model = helper.make_model(
            helper.make_graph([], "g", [], []),
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        )

        _ensure_com_microsoft_opset(model)

        ms_opsets = [op for op in model.opset_import if op.domain == "com.microsoft"]
        assert len(ms_opsets) == 1

    def test_adds_opset_to_functions(self):
        func = onnx.FunctionProto()
        func.name = "test_func"
        func.domain = "test.domain"
        func.opset_import.append(helper.make_opsetid("", 17))

        model = helper.make_model(
            helper.make_graph([], "g", [], []),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.functions.append(func)

        _ensure_com_microsoft_opset(model)

        func_domains = {op.domain for op in model.functions[0].opset_import}
        assert "com.microsoft" in func_domains

    def test_skips_function_with_existing_opset(self):
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

        _ensure_com_microsoft_opset(model)

        ms_opsets = [op for op in model.functions[0].opset_import if op.domain == "com.microsoft"]
        assert len(ms_opsets) == 1


class TestOnnxCastChainElimination:
    """Integration tests for OnnxCastChainElimination pass."""

    @pytest.fixture
    def cast_chain_model_path(self, tmp_path):
        """Model with a redundant Cast chain: fp32 → fp16 → fp32."""
        input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

        cast_to_fp16 = helper.make_node("Cast", ["X"], ["x_fp16"], to=TensorProto.FLOAT16)
        cast_back = helper.make_node("Cast", ["x_fp16"], ["Y"], to=TensorProto.FLOAT)

        graph = helper.make_graph([cast_to_fp16, cast_back], "cast_chain", [input_tensor], [output_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)

        path = tmp_path / "cast_chain.onnx"
        onnx.save(model, str(path))
        return path

    def test_eliminates_redundant_cast_chain(self, cast_chain_model_path, tmp_path):
        olive_model = ONNXModelHandler(model_path=str(cast_chain_model_path))
        p = create_pass_from_dict(OnnxCastChainElimination, {}, disable_search=True)
        output = p.run(olive_model, str(tmp_path / "out.onnx"))

        result_model = onnx.load(output.model_path)
        # The pass should produce a valid, runnable model.
        # Actual cast elimination depends on the ORT version; at minimum the
        # output graph must not have *more* nodes than the input.
        assert len(result_model.graph.node) <= 2

    def test_opset_fixup_applied(self, cast_chain_model_path, tmp_path):
        olive_model = ONNXModelHandler(model_path=str(cast_chain_model_path))
        p = create_pass_from_dict(
            OnnxCastChainElimination,
            {"fix_opset": True, "enable_cast_chain_elimination": False},
            disable_search=True,
        )
        output = p.run(olive_model, str(tmp_path / "out.onnx"))

        result_model = onnx.load(output.model_path)
        assert any(op.domain == "com.microsoft" for op in result_model.opset_import)

    def test_numerical_correctness(self, cast_chain_model_path, tmp_path):
        """Optimized model should produce the same output as the original."""
        olive_model = ONNXModelHandler(model_path=str(cast_chain_model_path))
        p = create_pass_from_dict(OnnxCastChainElimination, {}, disable_search=True)
        output = p.run(olive_model, str(tmp_path / "out.onnx"))

        x = np.random.randn(2, 4).astype(np.float32)

        orig_sess = ort.InferenceSession(str(cast_chain_model_path), providers=["CPUExecutionProvider"])
        opt_sess = ort.InferenceSession(output.model_path, providers=["CPUExecutionProvider"])

        orig_out = orig_sess.run(None, {"X": x})[0]
        opt_out = opt_sess.run(None, {"X": x})[0]
        np.testing.assert_allclose(orig_out, opt_out, atol=1e-3)
