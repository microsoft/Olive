# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access

import sys
from unittest.mock import MagicMock, patch

from olive.passes.onnx.discrepancy_check import _longest_common_token_sequence


class TestLongestCommonTokenSequence:
    """Unit tests for _longest_common_token_sequence helper."""

    def test_identical_sequences(self):
        assert _longest_common_token_sequence([1, 2, 3, 4], [1, 2, 3, 4]) == 4

    def test_empty_sequences(self):
        assert _longest_common_token_sequence([], []) == 0
        assert _longest_common_token_sequence([1, 2], []) == 0
        assert _longest_common_token_sequence([], [1, 2]) == 0

    def test_no_common_prefix(self):
        assert _longest_common_token_sequence([1, 2, 3], [4, 5, 6]) == 0

    def test_partial_common_prefix(self):
        assert _longest_common_token_sequence([1, 2, 3, 4, 5], [1, 2, 3, 9, 10]) == 3

    def test_one_is_prefix_of_other(self):
        assert _longest_common_token_sequence([1, 2, 3], [1, 2, 3, 4, 5]) == 3
        assert _longest_common_token_sequence([1, 2, 3, 4, 5], [1, 2, 3]) == 3

    def test_single_element_match(self):
        assert _longest_common_token_sequence([7], [7]) == 1

    def test_single_element_no_match(self):
        assert _longest_common_token_sequence([7], [8]) == 0

    def test_diverges_after_first(self):
        assert _longest_common_token_sequence([1, 99, 99], [1, 2, 3]) == 1

    def test_common_tokens_later_not_counted(self):
        # Tokens match later but not from the beginning
        assert _longest_common_token_sequence([10, 1, 2, 3], [20, 1, 2, 3]) == 0


class TestCompareGeneration:
    """Unit tests for OnnxDiscrepancyCheck.compare_generation."""

    def test_compare_generation_returns_common_prefix_length(self):
        """Test that compare_generation correctly computes the common prefix length."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        # Mock config
        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = "mock_genai_model"
        config.generate_prompt = "Hello world"
        config.generate_max_new_tokens = 10

        # Mock transformers tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))

        # Transformers generates [1, 2, 3, 10, 11, 12, 13]
        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11, 12, 13]])

        # GenAI generates [1, 2, 3, 10, 11, 99, 99] (diverges at index 5)
        mock_og = MagicMock()
        mock_genai_model = MagicMock()
        mock_og.Model.return_value = mock_genai_model
        mock_genai_tokenizer = MagicMock()
        mock_og.Tokenizer.return_value = mock_genai_tokenizer
        mock_genai_tokenizer.encode.return_value = [1, 2, 3]

        mock_params = MagicMock()
        mock_og.GeneratorParams.return_value = mock_params

        # Simulate generator producing tokens: 10, 11, 99, 99 then done
        mock_generator = MagicMock()
        genai_new_tokens = [10, 11, 99, 99]
        call_count = [0]

        def is_done_side_effect():
            return call_count[0] >= len(genai_new_tokens)

        def get_next_tokens_side_effect():
            token = genai_new_tokens[call_count[0]]
            call_count[0] += 1
            return [token]

        mock_generator.is_done = is_done_side_effect
        mock_generator.get_next_tokens = get_next_tokens_side_effect
        mock_og.Generator.return_value = mock_generator

        with (
            patch.dict(sys.modules, {"onnxruntime_genai": mock_og}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_generation(config, mock_ref_model)

        mock_generator.append_tokens.assert_called_once_with([[1, 2, 3]])
        # Common prefix: [1, 2, 3, 10, 11] = 5 tokens before divergence
        assert result == 5

    def test_compare_generation_fully_matching(self):
        """Test when both outputs are identical."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = "mock_genai_model"
        config.generate_prompt = "Test"
        config.generate_max_new_tokens = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[10, 20]]))

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50]])

        mock_og = MagicMock()
        mock_og.Model.return_value = MagicMock()
        mock_genai_tokenizer = MagicMock()
        mock_og.Tokenizer.return_value = mock_genai_tokenizer
        mock_genai_tokenizer.encode.return_value = [10, 20]

        mock_params = MagicMock()
        mock_og.GeneratorParams.return_value = mock_params

        mock_generator = MagicMock()
        genai_new_tokens = [30, 40, 50]
        call_count = [0]

        def is_done_side_effect():
            return call_count[0] >= len(genai_new_tokens)

        def get_next_tokens_side_effect():
            token = genai_new_tokens[call_count[0]]
            call_count[0] += 1
            return [token]

        mock_generator.is_done = is_done_side_effect
        mock_generator.get_next_tokens = get_next_tokens_side_effect
        mock_og.Generator.return_value = mock_generator

        with (
            patch.dict(sys.modules, {"onnxruntime_genai": mock_og}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_generation(config, mock_ref_model)

        mock_generator.append_tokens.assert_called_once_with([[10, 20]])
        # All 5 tokens match
        assert result == 5


class TestWeightDtypeInference:
    """Unit tests for ONNX weight dtype inference used to match the reference model precision."""

    @staticmethod
    def _build_tiny_llm_onnx(onnx_float_dtype):
        """Build a tiny-llm-like ONNX model whose float weights use the given dtype.

        The graph mimics a minimal language-model head: an embedding weight and a
        linear (lm_head) weight, plus an int64 buffer that must be ignored by the
        float dtype inference.
        """
        from onnx import TensorProto, helper

        hidden = 4
        vocab = 8

        embed_weight = helper.make_tensor("embed.weight", onnx_float_dtype, [vocab, hidden], [0.1] * (vocab * hidden))
        lm_head_weight = helper.make_tensor(
            "lm_head.weight", onnx_float_dtype, [hidden, vocab], [0.2] * (hidden * vocab)
        )
        # Non-float buffer that must be ignored when inferring the weight dtype.
        position_ids = helper.make_tensor("position_ids", TensorProto.INT64, [hidden], [0, 1, 2, 3])

        input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1])
        logits = helper.make_tensor_value_info("logits", onnx_float_dtype, [1, vocab])

        gather = helper.make_node("Gather", ["embed.weight", "input_ids"], ["hidden_states"])
        matmul = helper.make_node("MatMul", ["hidden_states", "lm_head.weight"], ["logits"])

        graph = helper.make_graph(
            [gather, matmul],
            "tiny_llm",
            [input_ids],
            [logits],
            initializer=[embed_weight, lm_head_weight, position_ids],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_infer_onnx_weight_dtype_float32(self):
        import onnx

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype

        model = self._build_tiny_llm_onnx(onnx.TensorProto.FLOAT)
        assert _infer_onnx_weight_dtype(model) == onnx.TensorProto.FLOAT

    def test_infer_onnx_weight_dtype_bfloat16(self):
        import onnx

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype

        model = self._build_tiny_llm_onnx(onnx.TensorProto.BFLOAT16)
        assert _infer_onnx_weight_dtype(model) == onnx.TensorProto.BFLOAT16

    def test_infer_onnx_weight_dtype_returns_none_without_float_weights(self):
        from onnx import TensorProto, helper

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype

        only_int = helper.make_tensor("buffer", TensorProto.INT64, [2], [1, 2])
        graph = helper.make_graph([], "no_float", [], [], initializer=[only_int])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        assert _infer_onnx_weight_dtype(model) is None

    def test_onnx_dtype_to_torch_float32(self):
        import onnx
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_dtype_to_torch

        assert _onnx_dtype_to_torch(onnx.TensorProto.FLOAT) == torch.float32

    def test_onnx_dtype_to_torch_bfloat16(self):
        import onnx
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_dtype_to_torch

        assert _onnx_dtype_to_torch(onnx.TensorProto.BFLOAT16) == torch.bfloat16

    def test_onnx_output_to_torch_bfloat16_from_uint16(self):
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_output_to_torch

        expected = torch.tensor([[1.5, -2.0]], dtype=torch.bfloat16)
        onnx_output = expected.view(torch.uint16).cpu().numpy()
        actual = _onnx_output_to_torch(onnx_output, torch.bfloat16)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, expected)

    def test_onnx_output_to_torch_keeps_uint16_for_integer_reference(self):
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_output_to_torch

        expected = torch.tensor([[123, 456]], dtype=torch.uint16)
        actual = _onnx_output_to_torch(expected.cpu().numpy(), torch.uint16)
        assert actual.dtype == torch.uint16
        assert torch.equal(actual, expected)

    def test_tiny_llm_reference_model_cast_to_weight_dtype(self):
        """End-to-end check that the reference model is cast to the ONNX weight dtype.

        Builds a tiny-llm ONNX model in float32 and bfloat16, then verifies the
        inferred torch dtype matches what the pass would cast the reference model to.
        """
        import onnx
        import torch

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype, _onnx_dtype_to_torch

        for onnx_dtype, expected_torch_dtype in (
            (onnx.TensorProto.FLOAT, torch.float32),
            (onnx.TensorProto.BFLOAT16, torch.bfloat16),
        ):
            model = self._build_tiny_llm_onnx(onnx_dtype)
            inferred = _infer_onnx_weight_dtype(model)
            assert _onnx_dtype_to_torch(inferred) == expected_torch_dtype


class TestSpeedupSettings:
    def test_timing_iterations_default_is_5(self):
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        pass_config = OnnxDiscrepancyCheck._default_config(None)
        assert pass_config["timing_iterations"].default_value == 5

    def test_measure_speedup_skips_when_timing_iterations_is_zero(self):
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
        ref_model = MagicMock()
        session = MagicMock()

        result = pass_instance._measure_speedup(
            ref_model=ref_model,
            session=session,
            dataloader=MagicMock(),
            io_config=MagicMock(),
            torch_device=MagicMock(),
            warmup_iterations=3,
            timing_iterations=0,
        )

        assert result is None
        ref_model.assert_not_called()
        session.run.assert_not_called()
